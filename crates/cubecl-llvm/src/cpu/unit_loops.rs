//! Cubes as loops over their units.
//!
//! A CPU kernel is compiled for one unit and then wrapped in a loop over the cubes of the range
//! its call is handed, and, inside it, a loop over the units of each cube. One thread runs every
//! unit of a cube, so a cube barrier costs a return to the unit loop rather than a meeting of
//! threads.
//!
//! Each barrier ends a region of the body. The units of a cube run in rounds: in each, every
//! unit still running resumes after the barrier it last reached, and runs until it reaches the
//! next barrier or the end of the kernel. A cube is done once a round ends with every unit
//! finished. Units that finish early simply take no part in later rounds, and units of one cube
//! may reach different barriers, as they may on a GPU.
//!
//! What a unit holds across a barrier is held per unit:
//! - a value computed before a barrier and used after it is recomputed where it is used when
//!   it derives from the unit's coordinates and constants alone, and otherwise kept in a slot
//!   per unit;
//! - a local variable written in one region and read in another becomes one variable per unit.
//!
//! Shared memory is the stack memory of the call, reused by each cube it runs in turn.
//!
//! The approach is that of CPU implementations of CUDA and OpenCL (Stratton et al., 2008;
//! Karrenberg and Hack, 2012; Jääskeläinen et al., 2015). Like the latter two, it works on the
//! control flow graph, so barriers may sit in any loop or branch.
//!
//! - Stratton, J. A., Stone, S. S. and Hwu, W. W. (2008). MCUDA: An efficient implementation
//!   of CUDA kernels for multi-core CPUs. Languages and Compilers for Parallel Computing,
//!   LNCS 5335, 16–30.
//! - Karrenberg, R. and Hack, S. (2012). Improving performance of OpenCL on CPUs. Compiler
//!   Construction, LNCS 7210, 1–20.
//! - Jääskeläinen, P. et al. (2015). pocl: A performance-portable OpenCL implementation.
//!   International Journal of Parallel Programming 43, 752–785.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    cpu::entrypoint::{
        ATTR_CUBE_BEGIN, ATTR_CUBE_COUNT_X, ATTR_CUBE_COUNT_Y, ATTR_CUBE_END, ATTR_UNIT_BEGIN,
        ATTR_UNIT_END, LOOP_BUILTINS, runtime_arg,
    },
    prelude::*,
};
use cubecl_core::{
    ir::{
        attributes::IndexAttr,
        dialect::{
            general::{ReadBuiltinOp, ReinterpretCastOp},
            memory::{AddressSpaceAttr, DeclareVariableOp, IndexOp, LoadOp, StoreOp},
            synchronization::{SyncOp, SyncScope},
        },
        interfaces::{MaterializableOp, side_effects::MemoryEffectsOp},
        types::ArrayType,
    },
    prelude::*,
};
use pliron::{linked_list::ContainsLinkedList, opts::dce::SideEffects};

/// Wraps an entry point in its cube and unit loops.
#[derive(Default)]
pub struct UnitLoopsPass;

#[pass_name]
impl Pass for UnitLoopsPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();
        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };
        let Some(abi) = func.get_entrypoint_abi(ctx) else {
            return Ok(res);
        };
        Kernel::new(ctx, func, abi.cube_dim).wrap(ctx);
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

/// What a unit computes, and where its barriers divide it.
struct Kernel {
    func: FuncOp,
    cube_dim: Dim3,
    /// The function's entry block: its arguments, the variables of the call and the start of
    /// the cube loop.
    setup: Ptr<BasicBlock>,
    /// Where a unit starts.
    body_entry: Ptr<BasicBlock>,
    /// Every block of the unit's body.
    body: Vec<Ptr<BasicBlock>>,
    /// Each barrier: the block that reaches it, and the block that resumes after it.
    barriers: Vec<(Ptr<BasicBlock>, Ptr<BasicBlock>)>,
    /// The coordinate reads the loops provide.
    placeholders: Vec<(Builtin, Ptr<Operation>)>,
}

/// The values the loops give each unit.
struct Unit {
    /// Where each unit starts, ahead of its body.
    dispatch: Ptr<BasicBlock>,
    /// The unit's index as a memory index, for the slots and variables it holds per unit.
    slot: Value,
    /// Whether any unit reached a barrier earlier in this round.
    alive: Option<Value>,
    /// Where a unit leaves its body for the next unit.
    latch: Ptr<BasicBlock>,
    /// Which barrier each unit resumes after.
    state: Option<Value>,
}

impl Kernel {
    fn new(ctx: &mut Context, func: FuncOp, cube_dim: Dim3) -> Self {
        let setup = func.get_entry_block(ctx);
        let body_entry = new_block(ctx, "unit_entry", vec![]);
        body_entry.insert_after(ctx, setup);
        let ops: Vec<_> = setup.deref(ctx).iter(ctx).collect();
        for op in ops {
            op.unlink(ctx);
            op.insert_at_back(body_entry, ctx);
        }

        let mut placeholders = Vec::new();
        let region = func.get_region(ctx);
        let mut body: Vec<_> = region
            .deref(ctx)
            .iter(ctx)
            .filter(|block| *block != setup)
            .collect();
        for &block in &body {
            for op in block.deref(ctx).iter(ctx) {
                if let Some(read) = op.as_op::<ReadBuiltinOp>(ctx) {
                    let builtin = read.builtin(ctx).0;
                    if LOOP_BUILTINS.contains(&builtin) {
                        placeholders.push((builtin, op));
                    }
                }
            }
        }

        let mut barriers = Vec::new();
        let mut pending: VecDeque<_> = body.iter().copied().collect();
        while let Some(block) = pending.pop_front() {
            let barrier = block
                .deref(ctx)
                .iter(ctx)
                .find(|op| is_cube_barrier(ctx, *op));
            let Some(barrier) = barrier else {
                continue;
            };
            let resume = new_block(ctx, "resume", vec![]);
            resume.insert_after(ctx, block);
            let after: Vec<_> = block
                .deref(ctx)
                .iter(ctx)
                .skip_while(|op| *op != barrier)
                .skip(1)
                .collect();
            for op in after {
                op.unlink(ctx);
                op.insert_at_back(resume, ctx);
            }
            Operation::erase(barrier, ctx);
            llvm::BrOp::new(ctx, resume, vec![])
                .get_operation()
                .insert_at_back(block, ctx);
            barriers.push((block, resume));
            body.push(resume);
            pending.push_back(resume);
        }

        Self {
            func,
            cube_dim,
            setup,
            body_entry,
            body,
            barriers,
            placeholders,
        }
    }

    fn wrap(self, ctx: &mut Context) {
        let crossing = Crossing::of(ctx, &self.body, &self.barriers);
        let declarations = self.declarations(ctx, &crossing);
        let held = self.held_values(ctx, &crossing, &declarations);

        let (unit, coordinates) = self.loops(ctx);

        for (barrier, (block, _)) in self.barriers.iter().enumerate() {
            let terminator = block.deref(ctx).get_terminator(ctx).unwrap();
            let state = unit.state.unwrap();
            emit_before(ctx, terminator, |scope| {
                let resume_code = u32::try_from(barrier + 1).unwrap();
                let code = constant::expand(scope, resume_code).value(scope);
                store_at(scope, state, unit.slot, code);
            });
            Operation::erase(terminator, ctx);
            let yes = emit_at_end(ctx, *block, |scope| yes::expand(scope).value(scope));
            llvm::BrOp::new(ctx, unit.latch, vec![yes])
                .get_operation()
                .insert_at_back(*block, ctx);
        }

        for &block in &self.body {
            let terminator = block.deref(ctx).get_terminator(ctx).unwrap();
            if terminator.as_op::<llvm::ReturnOp>(ctx).is_none() {
                continue;
            }
            if let Some(state) = unit.state {
                emit_before(ctx, terminator, |scope| {
                    let finished = constant::expand(scope, FINISHED).value(scope);
                    store_at(scope, state, unit.slot, finished);
                });
            }
            Operation::erase(terminator, ctx);
            let args = unit.alive.into_iter().collect();
            llvm::BrOp::new(ctx, unit.latch, args)
                .get_operation()
                .insert_at_back(block, ctx);
        }

        self.hold(ctx, &held, &unit);
        self.privatize(ctx, &declarations, &unit);

        for (builtin, read) in &self.placeholders {
            let value = coordinates[&builtin_key(*builtin)];
            let result = read.deref(ctx).get_result(0);
            result.replace_all_uses_with(ctx, &value);
            Operation::erase(*read, ctx);
        }
    }

    /// Builds the cube and unit loops around the body, and returns what they give each unit and
    /// the value of each coordinate.
    fn loops(&self, ctx: &mut Context) -> (Unit, HashMap<u8, Value>) {
        let func = self.func;
        let (dim_x, dim_y) = (self.cube_dim.x, self.cube_dim.y);
        let units = self.cube_dim.num_elems();
        let rounds = !self.barriers.is_empty();

        let cube_begin = runtime_arg(ctx, func, &ATTR_CUBE_BEGIN);
        let cube_end = runtime_arg(ctx, func, &ATTR_CUBE_END);
        let unit_begin = runtime_arg(ctx, func, &ATTR_UNIT_BEGIN);
        let unit_end = runtime_arg(ctx, func, &ATTR_UNIT_END);
        let count_x = runtime_arg(ctx, func, &ATTR_CUBE_COUNT_X);
        let count_y = runtime_arg(ctx, func, &ATTR_CUBE_COUNT_Y);

        let u32_ty = emit_at_end(ctx, self.setup, u32::__expand_as_type);
        let u64_ty = emit_at_end(ctx, self.setup, u64::__expand_as_type);
        let bool_ty = emit_at_end(ctx, self.setup, bool::__expand_as_type);

        let state = rounds.then(|| {
            let ty = ArrayType::get(ctx, u32_ty, units as usize).into();
            let state = DeclareVariableOp::new(
                ctx,
                TypeAttr::new(ty),
                AddressSpaceAttr(AddressSpace::Local),
                IndexAttr(4),
                None,
            );
            state.get_operation().insert_at_back(self.setup, ctx);
            state.get_result(ctx)
        });

        let cube_header = new_block(ctx, "cube_header", vec![u64_ty]);
        let cube_body = new_block(ctx, "cube_body", vec![]);
        let unit_args = if rounds {
            vec![u32_ty, bool_ty, bool_ty]
        } else {
            vec![u32_ty]
        };
        let unit_header = new_block(ctx, "unit_header", unit_args);
        let dispatch = new_block(ctx, "unit_dispatch", vec![]);
        let latch_args = if rounds { vec![bool_ty] } else { vec![] };
        let unit_latch = new_block(ctx, "unit_latch", latch_args);
        let round_end = new_block(ctx, "round_end", vec![]);
        let cube_latch = new_block(ctx, "cube_latch", vec![]);
        let cube_exit = new_block(ctx, "cube_exit", vec![]);

        let mut previous = self.setup;
        for block in [cube_header, cube_body, unit_header, dispatch] {
            block.insert_after(ctx, previous);
            previous = block;
        }
        let last = *self.body.last().unwrap_or(&self.body_entry);
        let last = self
            .func
            .get_region(ctx)
            .deref(ctx)
            .iter(ctx)
            .last()
            .unwrap_or(last);
        let mut previous = last;
        for block in [unit_latch, round_end, cube_latch, cube_exit] {
            block.insert_after(ctx, previous);
            previous = block;
        }

        // The call's range of cubes.
        llvm::BrOp::new(ctx, cube_header, vec![cube_begin])
            .get_operation()
            .insert_at_back(self.setup, ctx);
        let cube = cube_header.deref(ctx).get_argument(0);
        let more_cubes = emit_at_end(ctx, cube_header, |scope| {
            below_u64::expand(scope, cube.into(), cube_end.into()).value(scope)
        });
        llvm::CondBrOp::new(ctx, more_cubes, cube_body, vec![], cube_exit, vec![])
            .get_operation()
            .insert_at_back(cube_header, ctx);

        let mut coordinates = HashMap::new();
        let first_round = emit_at_end(ctx, cube_body, |scope| {
            coordinates.insert(
                builtin_key(Builtin::CubePosX),
                cube_x::expand(scope, cube.into(), count_x.into()).value(scope),
            );
            coordinates.insert(
                builtin_key(Builtin::CubePosY),
                cube_y::expand(scope, cube.into(), count_x.into(), count_y.into()).value(scope),
            );
            coordinates.insert(
                builtin_key(Builtin::CubePosZ),
                cube_z::expand(scope, cube.into(), count_x.into(), count_y.into()).value(scope),
            );
            rounds.then(|| {
                (
                    no::expand(scope).value(scope),
                    yes::expand(scope).value(scope),
                )
            })
        });
        let start_args = match first_round {
            Some((no, yes)) => vec![unit_begin, no, yes],
            None => vec![unit_begin],
        };
        llvm::BrOp::new(ctx, unit_header, start_args)
            .get_operation()
            .insert_at_back(cube_body, ctx);

        // The units of a cube, in rounds when it has barriers.
        let index = unit_header.deref(ctx).get_argument(0);
        let alive = rounds.then(|| unit_header.deref(ctx).get_argument(1));
        let first = rounds.then(|| unit_header.deref(ctx).get_argument(2));
        let more_units = emit_at_end(ctx, unit_header, |scope| {
            below::expand(scope, index.into(), unit_end.into()).value(scope)
        });
        let after_units = if rounds { round_end } else { cube_latch };
        llvm::CondBrOp::new(ctx, more_units, dispatch, vec![], after_units, vec![])
            .get_operation()
            .insert_at_back(unit_header, ctx);

        let slot = emit_at_end(ctx, dispatch, |scope| {
            coordinates.insert(
                builtin_key(Builtin::UnitPosX),
                unit_x::expand(scope, index.into(), dim_x).value(scope),
            );
            coordinates.insert(
                builtin_key(Builtin::UnitPosY),
                unit_y::expand(scope, index.into(), dim_x, dim_y).value(scope),
            );
            coordinates.insert(
                builtin_key(Builtin::UnitPosZ),
                unit_z::expand(scope, index.into(), dim_x, dim_y).value(scope),
            );
            as_index::expand(scope, index.into()).value(scope)
        });

        if let (Some(state), Some(first), Some(alive)) = (state, first, alive) {
            // The first round starts every unit at the top; later ones resume each after the
            // barrier it reached.
            let resume = new_block(ctx, "unit_resume", vec![]);
            resume.insert_after(ctx, dispatch);
            llvm::CondBrOp::new(ctx, first, self.body_entry, vec![], resume, vec![])
                .get_operation()
                .insert_at_back(dispatch, ctx);
            let reached = emit_at_end(ctx, resume, |scope| load_at(scope, state, slot));
            let mut test = resume;
            for (barrier, (_, after)) in self.barriers.iter().enumerate() {
                let code = u32::try_from(barrier + 1).unwrap();
                let is_here = emit_at_end(ctx, test, |scope| {
                    equals::expand(scope, reached.into(), code).value(scope)
                });
                let next = new_block(ctx, "unit_resume", vec![]);
                next.insert_after(ctx, test);
                llvm::CondBrOp::new(ctx, is_here, *after, vec![], next, vec![])
                    .get_operation()
                    .insert_at_back(test, ctx);
                test = next;
            }
            // A finished unit only passes on whether others are still running.
            llvm::BrOp::new(ctx, unit_latch, vec![alive])
                .get_operation()
                .insert_at_back(test, ctx);

            let still_running = unit_latch.deref(ctx).get_argument(0);
            let next = emit_at_end(ctx, unit_latch, |scope| {
                next_u32::expand(scope, index.into()).value(scope)
            });
            llvm::BrOp::new(ctx, unit_header, vec![next, still_running, first])
                .get_operation()
                .insert_at_back(unit_latch, ctx);

            let (no, no_again) = emit_at_end(ctx, round_end, |scope| {
                (
                    no::expand(scope).value(scope),
                    no::expand(scope).value(scope),
                )
            });
            llvm::CondBrOp::new(
                ctx,
                alive,
                unit_header,
                vec![unit_begin, no, no_again],
                cube_latch,
                vec![],
            )
            .get_operation()
            .insert_at_back(round_end, ctx);
        } else {
            llvm::BrOp::new(ctx, self.body_entry, vec![])
                .get_operation()
                .insert_at_back(dispatch, ctx);
            let next = emit_at_end(ctx, unit_latch, |scope| {
                next_u32::expand(scope, index.into()).value(scope)
            });
            llvm::BrOp::new(ctx, unit_header, vec![next])
                .get_operation()
                .insert_at_back(unit_latch, ctx);
            // Unused without barriers.
            llvm::UnreachableOp::new(ctx)
                .get_operation()
                .insert_at_back(round_end, ctx);
        }

        let next_cube = emit_at_end(ctx, cube_latch, |scope| {
            next_u64::expand(scope, cube.into()).value(scope)
        });
        llvm::BrOp::new(ctx, cube_header, vec![next_cube])
            .get_operation()
            .insert_at_back(cube_latch, ctx);
        llvm::ReturnOp::new(ctx, None)
            .get_operation()
            .insert_at_back(cube_exit, ctx);

        let unit = Unit {
            dispatch,
            slot,
            alive,
            latch: unit_latch,
            state,
        };
        (unit, coordinates)
    }

    /// The local variables to hold per unit, and the shared memory of the call.
    fn declarations(&self, ctx: &Context, crossing: &Crossing) -> Declarations {
        let mut declarations = Declarations::default();
        for &block in &self.body {
            for op in block.deref(ctx).iter(ctx) {
                let Some(declare) = op.as_op::<DeclareVariableOp>(ctx) else {
                    continue;
                };
                match declare.addr_space(ctx).0 {
                    AddressSpace::Shared => declarations.shared.push(op),
                    AddressSpace::Local => {
                        let access = accesses(ctx, declare.get_result(ctx));
                        let mut blocks = access.blocks.clone();
                        blocks.insert(block);
                        let crosses = blocks
                            .iter()
                            .any(|from| access.blocks.iter().any(|to| crossing.after(*from, *to)));
                        if crosses && (access.written || access.escapes) {
                            declarations.private.push(op);
                        }
                    }
                    _ => {}
                }
            }
        }
        declarations
    }

    /// The values a unit holds across a barrier, and how.
    fn held_values(
        &self,
        ctx: &Context,
        crossing: &Crossing,
        declarations: &Declarations,
    ) -> Vec<(Value, Held, Vec<Use<Value>>)> {
        let mut available: HashSet<Value> = self.setup.deref(ctx).arguments().collect();
        for (_, read) in &self.placeholders {
            available.insert(read.deref(ctx).get_result(0));
        }
        for op in declarations.private.iter().chain(&declarations.shared) {
            available.insert(op.deref(ctx).get_result(0));
        }
        let mut recomputable = HashMap::new();

        let mut held = Vec::new();
        for &block in &self.body {
            let mut defined: Vec<Value> = block.deref(ctx).arguments().collect();
            for op in block.deref(ctx).iter(ctx) {
                defined.extend(op.deref(ctx).results());
            }
            for value in defined {
                if available.contains(&value) {
                    continue;
                }
                let crossed: Vec<_> = value
                    .uses(ctx)
                    .into_iter()
                    .filter(|r#use| {
                        let user = r#use.user_op().deref(ctx).get_parent_block().unwrap();
                        user != block && crossing.after(block, user)
                    })
                    .collect();
                if crossed.is_empty() {
                    continue;
                }
                let how = if recompute(ctx, value, &available, &mut recomputable, 0) {
                    Held::Recomputed
                } else {
                    Held::Slot
                };
                held.push((value, how, crossed));
            }
        }
        held
    }

    /// Recomputes or reloads each held value where it is used after a barrier.
    fn hold(&self, ctx: &mut Context, held: &[(Value, Held, Vec<Use<Value>>)], unit: &Unit) {
        let units = self.cube_dim.num_elems() as usize;
        for (value, how, uses) in held {
            match how {
                Held::Recomputed => {
                    for r#use in uses {
                        let user = r#use.user_op();
                        let operand = r#use.find_index(ctx);
                        let copy = recompute_before(ctx, *value, user);
                        Operation::replace_operand(user, ctx, operand, copy);
                    }
                }
                Held::Slot => {
                    let ty = value.get_type(ctx);
                    let align = alignment(ctx, ty);
                    let slots = ArrayType::get(ctx, ty, units).into();
                    let slots = DeclareVariableOp::new(
                        ctx,
                        TypeAttr::new(slots),
                        AddressSpaceAttr(AddressSpace::Local),
                        IndexAttr(align),
                        None,
                    );
                    let first_setup_op = self.setup.deref(ctx).get_terminator(ctx).unwrap();
                    slots.get_operation().insert_before(ctx, first_setup_op);
                    let slots = slots.get_result(ctx);

                    let store = |scope: &Scope| store_at(scope, slots, unit.slot, *value);
                    match value.defining_op() {
                        Some(def) => emit_after(ctx, def, store),
                        None => {
                            let block = value.defining_block().unwrap();
                            let first = block.deref(ctx).iter(ctx).next();
                            match first {
                                Some(first) => emit_before(ctx, first, store),
                                None => emit_at_end(ctx, block, store),
                            }
                        }
                    }
                    for r#use in uses {
                        let user = r#use.user_op();
                        let operand = r#use.find_index(ctx);
                        let loaded =
                            emit_before(ctx, user, |scope| load_at(scope, slots, unit.slot));
                        Operation::replace_operand(user, ctx, operand, loaded);
                    }
                }
            }
        }
    }

    /// Gives each unit its own copy of the variables it holds across a barrier, and makes the
    /// shared memory the call's own.
    fn privatize(&self, ctx: &mut Context, declarations: &Declarations, unit: &Unit) {
        let units = self.cube_dim.num_elems() as usize;
        let setup_end = self.setup.deref(ctx).get_terminator(ctx).unwrap();
        let dispatch_end = unit.dispatch.deref(ctx).get_terminator(ctx).unwrap();

        for &op in &declarations.private {
            let declare = op.as_op::<DeclareVariableOp>(ctx).unwrap();
            let ty = declare.value_ty(ctx).get_type(ctx);
            let align = declare.alignment(ctx).0;
            let per_unit = ArrayType::get(ctx, ty, units).into();
            let per_unit = DeclareVariableOp::new(
                ctx,
                TypeAttr::new(per_unit),
                AddressSpaceAttr(AddressSpace::Local),
                IndexAttr(align),
                None,
            );
            per_unit.get_operation().insert_before(ctx, setup_end);
            let per_unit = per_unit.get_result(ctx);
            let own = emit_before(ctx, dispatch_end, |scope| {
                let index = IndexOp::new(scope.ctx_mut(), per_unit, unit.slot, None);
                scope.register(&index);
                index.get_result(scope.ctx())
            });
            let original = declare.get_result(ctx);
            if declare.initializer(ctx).is_some() {
                // The declaration stays where it was, to initialize the unit's copy each time
                // the unit reaches it.
                let load = emit_after(ctx, op, |scope| {
                    let load = LoadOp::new(scope.ctx_mut(), original);
                    scope.register(&load);
                    let value = load.get_result(scope.ctx());
                    let store = StoreOp::new(scope.ctx_mut(), own, value);
                    scope.register(&store);
                    load.get_operation()
                });
                original.replace_some_uses_with(ctx, |_, r#use| r#use.user_op() != load, &own);
            } else {
                original.replace_all_uses_with(ctx, &own);
                Operation::erase(op, ctx);
            }
        }

        for &op in &declarations.shared {
            let declare = op.as_op::<DeclareVariableOp>(ctx).unwrap();
            let ty = declare.value_ty(ctx).clone();
            let align = *declare.alignment(ctx);
            let shared_ty = declare.get_result(ctx).get_type(ctx);
            let local =
                DeclareVariableOp::new(ctx, ty, AddressSpaceAttr(AddressSpace::Local), align, None);
            local.get_operation().insert_before(ctx, setup_end);
            let local = local.get_result(ctx);
            let cast = ReinterpretCastOp::new(ctx, shared_ty, local);
            cast.get_operation().insert_before(ctx, setup_end);
            declare
                .get_result(ctx)
                .replace_all_uses_with(ctx, &cast.get_result(ctx));
            Operation::erase(op, ctx);
        }
    }
}

/// The resume code of a unit that has finished.
const FINISHED: u32 = 0;

enum Held {
    /// Recomputed from the unit's coordinates and constants where it is used.
    Recomputed,
    /// Kept in a slot per unit.
    Slot,
}

#[derive(Default)]
struct Declarations {
    private: Vec<Ptr<Operation>>,
    shared: Vec<Ptr<Operation>>,
}

/// Where a variable's memory is reached, and how.
struct Accesses {
    blocks: HashSet<Ptr<BasicBlock>>,
    written: bool,
    /// Its address leaves for another block or for memory, where it can no longer be followed.
    escapes: bool,
}

fn accesses(ctx: &Context, pointer: Value) -> Accesses {
    let mut access = Accesses {
        blocks: HashSet::new(),
        written: false,
        escapes: false,
    };
    let mut pending = vec![pointer];
    while let Some(pointer) = pending.pop() {
        for r#use in pointer.uses(ctx) {
            let user = r#use.user_op();
            access
                .blocks
                .insert(user.deref(ctx).get_parent_block().unwrap());
            if user.as_op::<LoadOp>(ctx).is_some() {
                continue;
            }
            if let Some(index) = user.as_op::<IndexOp>(ctx)
                && index.base(ctx) == pointer
            {
                pending.push(index.get_result(ctx));
                continue;
            }
            if user.deref(ctx).get_num_successors() > 0 {
                access.escapes = true;
                continue;
            }
            if let Some(store) = user.as_op::<StoreOp>(ctx)
                && store.value(ctx) == pointer
            {
                access.escapes = true;
            }
            access.written = true;
        }
    }
    access
}

/// Which blocks a unit can reach having passed a barrier, from each block.
struct Crossing {
    after: HashMap<Ptr<BasicBlock>, HashSet<Ptr<BasicBlock>>>,
}

impl Crossing {
    fn of(
        ctx: &Context,
        body: &[Ptr<BasicBlock>],
        barriers: &[(Ptr<BasicBlock>, Ptr<BasicBlock>)],
    ) -> Self {
        let reach: HashMap<_, _> = body
            .iter()
            .map(|&block| (block, reachable(ctx, block)))
            .collect();
        let after = body
            .iter()
            .map(|&block| {
                let mut after = HashSet::new();
                for (reaching, resume) in barriers {
                    if reach[&block].contains(reaching) {
                        after.extend(reach[resume].iter().copied());
                    }
                }
                (block, after)
            })
            .collect();
        Self { after }
    }

    /// Whether a unit can reach `to` from `from` by way of a barrier.
    fn after(&self, from: Ptr<BasicBlock>, to: Ptr<BasicBlock>) -> bool {
        self.after
            .get(&from)
            .is_some_and(|after| after.contains(&to))
    }
}

/// `block` and every block reachable from it.
fn reachable(ctx: &Context, block: Ptr<BasicBlock>) -> HashSet<Ptr<BasicBlock>> {
    let mut seen = HashSet::from([block]);
    let mut pending = vec![block];
    while let Some(block) = pending.pop() {
        for succ in block.deref(ctx).succs(ctx) {
            if seen.insert(succ) {
                pending.push(succ);
            }
        }
    }
    seen
}

/// Whether `value` can be recomputed wherever it is used, from values available to every unit:
/// a pure operation of such values.
fn recompute(
    ctx: &Context,
    value: Value,
    available: &HashSet<Value>,
    known: &mut HashMap<Value, bool>,
    depth: u32,
) -> bool {
    if available.contains(&value) {
        return true;
    }
    if let Some(&answer) = known.get(&value) {
        return answer;
    }
    let answer = depth < 16
        && value.defining_op().is_some_and(|op| {
            pure(ctx, op)
                && op
                    .deref(ctx)
                    .operands()
                    .all(|operand| recompute(ctx, operand, available, known, depth + 1))
        });
    known.insert(value, answer);
    answer
}

fn pure(ctx: &Context, op: Ptr<Operation>) -> bool {
    let operation = op.deref(ctx);
    if operation.num_regions() > 0 || operation.get_num_successors() > 0 {
        return false;
    }
    drop(operation);
    if op.as_op::<ConstantOp>(ctx).is_some() {
        return true;
    }
    if op.as_op::<DeclareVariableOp>(ctx).is_some() {
        return false;
    }
    let dyn_op = op.dyn_op(ctx);
    let no_side_effects =
        op_cast::<dyn SideEffects>(&*dyn_op).is_some_and(|effects| !effects.has_side_effects(ctx));
    let no_memory_effects =
        op_cast::<dyn MemoryEffectsOp>(&*dyn_op).is_some_and(|effects| !effects.has_effects(ctx));
    let materializable = op_cast::<dyn MaterializableOp>(&*dyn_op).is_some();
    no_side_effects && no_memory_effects && materializable
}

/// A copy of the computation of `value`, placed before `user`.
fn recompute_before(ctx: &mut Context, value: Value, user: Ptr<Operation>) -> Value {
    let Some(op) = value.defining_op() else {
        return value;
    };
    if !pure(ctx, op) {
        return value;
    }
    let operands: Vec<Value> = op.deref(ctx).operands().collect();
    let operands = operands
        .into_iter()
        .map(|operand| {
            if needs_copy(ctx, operand, user) {
                recompute_before(ctx, operand, user)
            } else {
                operand
            }
        })
        .collect();
    let index = value.find_index(ctx);
    let result_types = op.deref(ctx).results().map(|r| r.get_type(ctx)).collect();
    let attributes = op.deref(ctx).attributes.clone();
    let dyn_op = op.dyn_op(ctx);
    let materialize = op_cast::<dyn MaterializableOp>(&*dyn_op).unwrap();
    let copy = materialize.materialize(ctx, result_types, operands, attributes);
    copy.insert_before(ctx, user);
    copy.deref(ctx).get_result(index)
}

/// Whether `operand`, as `user` reads it, is computed in the unit's body rather than given to
/// every unit by the loops or the call.
fn needs_copy(ctx: &Context, operand: Value, user: Ptr<Operation>) -> bool {
    let user_block = user.deref(ctx).get_parent_block();
    let def_block = match operand.defining_op() {
        Some(op) => op.deref(ctx).get_parent_block(),
        None => operand.defining_block(),
    };
    let placeholder = operand
        .defining_op()
        .and_then(|op| op.as_op::<ReadBuiltinOp>(ctx))
        .is_some_and(|read| LOOP_BUILTINS.contains(&read.builtin(ctx).0));
    def_block != user_block && !placeholder && !is_function_argument(ctx, operand)
}

fn is_function_argument(ctx: &Context, value: Value) -> bool {
    value.defining_block().is_some_and(|block| {
        block
            .deref(ctx)
            .get_parent_op(ctx)
            .and_then(|op| op.as_op::<FuncOp>(ctx))
            .is_some_and(|func| func.get_entry_block(ctx) == block)
    })
}

fn alignment(ctx: &Context, ty: TypeHandle) -> usize {
    let ty = ty.deref(ctx);
    type_cast::<dyn AlignedType>(&*ty).map_or(8, |aligned| aligned.align(ctx))
}

fn is_cube_barrier(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.as_op::<SyncOp>(ctx)
        .is_some_and(|sync| sync.scope(ctx).0 == SyncScope::Cube)
}

fn builtin_key(builtin: Builtin) -> u8 {
    builtin as u8
}

fn new_block(ctx: &mut Context, label: &str, args: Vec<TypeHandle>) -> Ptr<BasicBlock> {
    BasicBlock::new(ctx, Some(label.try_into().unwrap()), args)
}

fn emit_at_end<R>(ctx: &mut Context, block: Ptr<BasicBlock>, f: impl FnOnce(&Scope) -> R) -> R {
    let mut inserter = OpInserter::new_at_block_end(block);
    let scope = Scope::from_context_and_inserter(ctx, &mut inserter);
    f(&scope)
}

fn emit_before<R>(ctx: &mut Context, op: Ptr<Operation>, f: impl FnOnce(&Scope) -> R) -> R {
    let mut inserter = OpInserter::new_before_operation(op);
    let scope = Scope::from_context_and_inserter(ctx, &mut inserter);
    f(&scope)
}

fn emit_after<R>(ctx: &mut Context, op: Ptr<Operation>, f: impl FnOnce(&Scope) -> R) -> R {
    let mut inserter = OpInserter::new_after_operation(op);
    let scope = Scope::from_context_and_inserter(ctx, &mut inserter);
    f(&scope)
}

fn store_at(scope: &Scope, base: Value, index: Value, value: Value) {
    let element = IndexOp::new(scope.ctx_mut(), base, index, None);
    scope.register(&element);
    let element = element.get_result(scope.ctx());
    let store = StoreOp::new(scope.ctx_mut(), element, value);
    scope.register(&store);
}

fn load_at(scope: &Scope, base: Value, index: Value) -> Value {
    let element = IndexOp::new(scope.ctx_mut(), base, index, None);
    scope.register(&element);
    let element = element.get_result(scope.ctx());
    let load = LoadOp::new(scope.ctx_mut(), element);
    scope.register(&load);
    load.get_result(scope.ctx())
}

#[cube]
fn constant(#[comptime] value: u32) -> u32 {
    value
}

#[cube]
fn yes() -> bool {
    true
}

#[cube]
fn no() -> bool {
    false
}

#[cube]
fn below(a: u32, b: u32) -> bool {
    a < b
}

#[cube]
fn below_u64(a: u64, b: u64) -> bool {
    a < b
}

#[cube]
fn equals(a: u32, #[comptime] b: u32) -> bool {
    a == b
}

#[cube]
fn next_u32(a: u32) -> u32 {
    a + 1
}

#[cube]
fn next_u64(a: u64) -> u64 {
    a + 1
}

#[cube]
fn as_index(a: u32) -> usize {
    a as usize
}

#[cube]
fn cube_x(cube: u64, count_x: u32) -> u32 {
    (cube % count_x as u64) as u32
}

#[cube]
fn cube_y(cube: u64, count_x: u32, count_y: u32) -> u32 {
    ((cube / count_x as u64) % count_y as u64) as u32
}

#[cube]
fn cube_z(cube: u64, count_x: u32, count_y: u32) -> u32 {
    (cube / (count_x as u64 * count_y as u64)) as u32
}

#[cube]
fn unit_x(unit: u32, #[comptime] dim_x: u32) -> u32 {
    unit % dim_x
}

#[cube]
fn unit_y(unit: u32, #[comptime] dim_x: u32, #[comptime] dim_y: u32) -> u32 {
    (unit / dim_x) % dim_y
}

#[cube]
fn unit_z(unit: u32, #[comptime] dim_x: u32, #[comptime] dim_y: u32) -> u32 {
    unit / (dim_x * dim_y)
}
