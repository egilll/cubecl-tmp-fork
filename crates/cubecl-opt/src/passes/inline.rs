//! Inlining of device function calls.
//!
//! A `#[cube]` function traced once into its own [`FuncOp`] is called with a
//! [`CallOp`]. This pass decides, call by call, whether the call survives to
//! the target or the callee's body replaces it, and then deletes private
//! functions nothing refers to any more.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use cubecl_environment::collections::HashSet;
use cubecl_ir::{
    dialect::{
        branch::ReturnOp,
        call::{CallOp, resolve_callee},
    },
    prelude::*,
};
use pliron::{
    builtin::{
        op_interfaces::{SymbolOpInterface, SymbolUserOpInterface},
        ops::{FuncOp, ModuleOp},
    },
    identifier::Identifier,
    input_error_noloc,
    irbuild::cloning::{IrMapping, clone_operation},
    linked_list::ContainsLinkedList,
};

/// Which calls the [`InlinePass`] replaces with the callee's body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InlinePolicy {
    /// Every call: for targets that cannot emit calls, and the behaviour of a
    /// kernel traced without device functions.
    #[default]
    All,
    /// Keep a call, and let the target's own compiler decide, unless
    /// inlining it is required or free:
    ///
    /// - required: the callee uses something only the kernel's own body can
    ///   (builtins, kernel metadata, synchronization, plane operations,
    ///   atomics, matrix or TMA operations, inline assembly, printing);
    /// - free: the callee has at most `max_inline_ops` operations, or a single
    ///   call site, which leaves it with no other user.
    Target {
        /// Callees this small are inlined at every call.
        max_inline_ops: usize,
    },
}

impl InlinePolicy {
    /// [`InlinePolicy::Target`] with the default size threshold: about the
    /// size of a call's own setup, so an inlined body costs no more than the
    /// call it replaces.
    pub const fn target() -> Self {
        Self::Target { max_inline_ops: 16 }
    }
}

/// How many times a body may be inlined into itself through nested calls
/// before the pass gives up. Device functions are not recursive, so reaching
/// it means the module is wrong rather than large.
const MAX_INLINE_DEPTH: usize = 64;

/// Replaces calls with their callee's body, according to an
/// [`InlinePolicy`], then deletes the private functions left without users.
/// A module-level pass: it runs before the per-function pipeline, so every
/// function pass sees the code it will emit.
#[derive(Default)]
pub struct InlinePass {
    pub policy: InlinePolicy,
}

impl InlinePass {
    pub fn new(policy: InlinePolicy) -> Self {
        Self { policy }
    }

    fn should_inline(&self, ctx: &Context, call: &CallOp, round: &mut Round) -> bool {
        let max_inline_ops = match self.policy {
            InlinePolicy::All => return true,
            InlinePolicy::Target { max_inline_ops } => max_inline_ops,
        };
        let symbol = call.callee_symbol(ctx);
        let Some(callee) = resolve_callee(ctx, call) else {
            // Inlining reports the missing callee.
            return true;
        };
        let decision = round.decisions.entry(symbol.clone()).or_insert_with(|| {
            if let Some(op) = entry_only_op(ctx, callee) {
                Decision::Inline(Reason::Requires(op))
            } else if round.call_sites.get(&symbol).copied().unwrap_or(0) <= 1 {
                Decision::Inline(Reason::SingleCallSite)
            } else {
                let ops = cubecl_ir::dialect::base::count_ops(ctx, callee.get_operation());
                match ops <= max_inline_ops {
                    true => Decision::Inline(Reason::Small(ops)),
                    false => Decision::Keep(ops),
                }
            }
        });
        log::debug!("inline `{symbol}`: {decision:?}");
        matches!(decision, Decision::Inline(_))
    }
}

/// What the pass knows about the module during one round of inlining.
#[derive(Default)]
struct Round {
    /// Calls to each function.
    call_sites: hashbrown::HashMap<Identifier, usize>,
    /// The decision for each callee, made once per round.
    decisions: hashbrown::HashMap<Identifier, Decision>,
}

#[allow(dead_code, reason = "read through Debug, as inlining remarks")]
#[derive(Debug)]
enum Decision {
    Inline(Reason),
    /// Kept as a call, with the callee's size in operations.
    Keep(usize),
}

#[allow(dead_code, reason = "read through Debug, as inlining remarks")]
#[derive(Debug)]
enum Reason {
    /// The callee uses an operation only the kernel's own body can.
    Requires(String),
    /// The callee has this many operations, no more than a call costs.
    Small(usize),
    /// The callee has no other caller.
    SingleCallSite,
}

/// The first operation in `func` that a function other than the kernel
/// can't contain, by name.
fn entry_only_op(ctx: &Context, func: FuncOp) -> Option<String> {
    let mut state = (func.get_operation(), None);
    pliron::graph::walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut state,
        &WALKCONFIG_PREORDER_FORWARD,
        func.get_operation(),
        |ctx, (func, found), node| {
            if found.is_some() {
                return;
            }
            if let IRNode::Operation(op) = node
                && op != *func
            {
                let name = Operation::get_opid(op, ctx).to_string();
                if !callable_op(&name) {
                    *found = Some(name);
                }
            }
        },
    );
    state.1
}

/// Whether an operation, by its `dialect.name`, can appear in a device
/// function: computation, control flow and the function's own local memory,
/// and nothing that reads the kernel's launch state or synchronizes with
/// other units.
fn callable_op(name: &str) -> bool {
    let (dialect, op) = name.split_once('.').unwrap_or((name, ""));
    match dialect {
        "math" | "cmp" | "bitwise" | "vector" | "composite" | "branch" | "builtin" => true,
        "memory" => op != "address_space",
        "cube" => matches!(
            op,
            "bool_and"
                | "bool_not"
                | "bool_or"
                | "cast"
                | "comment"
                | "copy"
                | "poison"
                | "reinterpret_cast"
                | "select"
        ),
        _ => false,
    }
}

#[pass_name]
impl Pass for InlinePass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        // Inlining a body brings its own calls along, so repeat until a round
        // inlines nothing. Each round goes one level deeper into the call
        // graph.
        for _ in 0..=MAX_INLINE_DEPTH {
            let calls = collect_calls(ctx, op);
            let mut round = Round::default();
            for call in &calls {
                *round.call_sites.entry(call.callee_symbol(ctx)).or_default() += 1;
            }
            let mut inlined = false;
            for call in calls {
                if self.should_inline(ctx, &call, &mut round) {
                    inline_call(ctx, &call)?;
                    inlined = true;
                }
            }
            if !inlined {
                if erase_unused_functions(ctx, op) {
                    res.ir_changed |= IRStatus::Changed;
                }
                if order_callees_first(ctx, op) {
                    res.ir_changed |= IRStatus::Changed;
                }
                return Ok(res);
            }
            res.ir_changed |= IRStatus::Changed;
        }

        Err(input_error_noloc!(InlineError::TooDeep(MAX_INLINE_DEPTH)))
    }
}

#[derive(Debug, thiserror::Error)]
enum InlineError {
    #[error(
        "calls are still nested {0} levels deep after inlining; is a device function recursive?"
    )]
    TooDeep(usize),
    #[error("the call to `{0}` names no function in the module")]
    UnknownCallee(String),
    #[error("`{0}` has {1} blocks; only a single-block body can be inlined")]
    MultipleBlocks(String, usize),
    #[error("`{0}` does not end with a return")]
    NoReturn(String),
}

fn collect_calls(ctx: &Context, op: Ptr<Operation>) -> Vec<CallOp> {
    let mut calls = Vec::new();
    pliron::graph::walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut calls,
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx, calls, node| {
            if let IRNode::Operation(op) = node
                && let Some(call) = Operation::get_op::<CallOp>(op, ctx)
            {
                calls.push(call);
            }
        },
    );
    calls
}

/// Replace `call` with a copy of its callee's body: the callee's arguments
/// map to the call's operands, and the call's results to the values the
/// callee returns.
fn inline_call(ctx: &mut Context, call: &CallOp) -> Result<()> {
    let call_op = call.get_operation();
    let symbol = call.callee_symbol(ctx);
    let Some(callee) = resolve_callee(ctx, call) else {
        return Err(input_error_noloc!(InlineError::UnknownCallee(
            symbol.to_string()
        )));
    };

    let region = callee.get_region(ctx);
    let num_blocks = region.deref(ctx).iter(ctx).count();
    if num_blocks != 1 {
        return Err(input_error_noloc!(InlineError::MultipleBlocks(
            symbol.to_string(),
            num_blocks
        )));
    }
    let body = callee.get_entry_block(ctx);

    let mut mapping = IrMapping::new();
    let params: Vec<Value> = body.deref(ctx).arguments().collect();
    let args: Vec<Value> = call_op.deref(ctx).operands().collect();
    for (param, arg) in params.into_iter().zip(args) {
        mapping.map_value(param, arg);
    }

    let ops: Vec<Ptr<Operation>> = body.deref(ctx).iter(ctx).collect();
    let Some((terminator, ops)) = ops.split_last() else {
        return Err(input_error_noloc!(InlineError::NoReturn(
            symbol.to_string()
        )));
    };
    let Some(ret) = Operation::get_op::<ReturnOp>(*terminator, ctx) else {
        return Err(input_error_noloc!(InlineError::NoReturn(
            symbol.to_string()
        )));
    };

    let mut rewriter = PassRewriter::default();
    for op in ops {
        let clone = clone_operation(*op, ctx, &mut rewriter, &mut mapping);
        clone.insert_before(ctx, call_op);
    }

    let returned: Vec<Value> = ret
        .get_operation()
        .deref(ctx)
        .operands()
        .map(|value| mapping.lookup_value_or_default(value))
        .collect();
    let results: Vec<Value> = call_op.deref(ctx).results().collect();
    for (result, value) in results.iter().zip(returned) {
        result.replace_all_uses_with(ctx, &value);
    }
    Operation::erase(call_op, ctx);
    Ok(())
}

/// Delete the private functions no operation in the module refers to.
/// Returns whether any was deleted.
fn erase_unused_functions(ctx: &mut Context, module: Ptr<Operation>) -> bool {
    let Some(module) = Operation::get_op::<ModuleOp>(module, ctx) else {
        return false;
    };
    let mut used: HashSet<Identifier> = HashSet::default();
    pliron::graph::walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut used,
        &WALKCONFIG_PREORDER_FORWARD,
        module.get_operation(),
        |ctx, used, node| {
            if let IRNode::Operation(op) = node
                && let Some(user) = op_cast::<dyn SymbolUserOpInterface>(&*op.dyn_op(ctx))
            {
                used.extend(user.used_symbols(ctx));
            }
        },
    );

    let body = module.get_body(ctx, 0);
    let unused: Vec<Ptr<Operation>> = body
        .deref(ctx)
        .iter(ctx)
        .filter(|op| {
            Operation::get_op::<FuncOp>(*op, ctx).is_some_and(|func| {
                let private =
                    !cubecl_ir::interfaces::control_flow::SymbolOpInterface::is_public(&func, ctx);
                private && !used.contains(&func.get_symbol_name(ctx))
            })
        })
        .collect();
    let erased = !unused.is_empty();
    for op in unused {
        Operation::erase(op, ctx);
    }
    erased
}

/// Move the module's functions so every function comes after the ones it
/// calls, with the kernel last: the order a C-like target needs to print
/// them without declaring them first. Returns whether anything moved.
fn order_callees_first(ctx: &mut Context, module: Ptr<Operation>) -> bool {
    let Some(module) = Operation::get_op::<ModuleOp>(module, ctx) else {
        return false;
    };
    let body = module.get_body(ctx, 0);
    let funcs: Vec<FuncOp> = body
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| Operation::get_op::<FuncOp>(op, ctx))
        .collect();
    if funcs.len() < 2 {
        return false;
    }

    // Depth-first from each function, emitting callees before callers.
    let mut ordered: Vec<FuncOp> = Vec::new();
    let mut visited: HashSet<Identifier> = HashSet::default();
    fn visit(
        ctx: &Context,
        func: FuncOp,
        funcs: &[FuncOp],
        visited: &mut HashSet<Identifier>,
        ordered: &mut Vec<FuncOp>,
    ) {
        if !visited.insert(func.get_symbol_name(ctx)) {
            return;
        }
        for call in collect_calls(ctx, func.get_operation()) {
            let symbol = call.callee_symbol(ctx);
            if let Some(callee) = funcs.iter().find(|f| f.get_symbol_name(ctx) == symbol) {
                visit(ctx, *callee, funcs, visited, ordered);
            }
        }
        ordered.push(func);
    }
    let (entries, others): (Vec<FuncOp>, Vec<FuncOp>) = funcs.iter().partition(|func| {
        cubecl_ir::attributes::EntrypointInterface::get_entrypoint_abi(*func, ctx).is_some()
    });
    for func in others.iter().chain(entries.iter()) {
        visit(ctx, *func, &funcs, &mut visited, &mut ordered);
    }

    if ordered == funcs {
        return false;
    }
    // Move every function, in order, to the end of the module: whatever
    // else the module holds keeps its place ahead of them.
    for func in ordered {
        let op = func.get_operation();
        op.unlink(ctx);
        op.insert_at_back(body, ctx);
    }
    true
}
