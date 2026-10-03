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

    fn should_inline(&self, _ctx: &Context, _call: &CallOp) -> bool {
        match self.policy {
            InlinePolicy::All => true,
        }
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
            let mut inlined = false;
            for call in calls {
                if self.should_inline(ctx, &call) {
                    inline_call(ctx, &call)?;
                    inlined = true;
                }
            }
            if !inlined {
                if erase_unused_functions(ctx, op) {
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
