//! Calls to device functions.
//!
//! A `#[cube]` function can be traced once into a private [`FuncOp`] in the
//! kernel module and called from each of its call sites, rather than traced
//! again into every caller. Whether a call survives to the target, or is
//! inlined, is the inliner's decision.

use alloc::{format, string::ToString};
use core::ops::Range;

use pliron::{
    builtin::{attributes::IdentifierAttr, ops::FuncOp, types::FunctionType},
    identifier::Identifier,
    printable::Printable,
    symbol_table::SymbolTableCollection,
    verify_err,
};

use crate::{
    HasSideEffects,
    dialect::general::SymbolUserOpVerifyErr,
    interfaces::{
        control_flow,
        side_effects::{MemoryEffect, MemoryEffectsOp},
    },
    prelude::*,
};

/// A direct call to a device function in the same module.
///
/// Its effects are unknown to the analyses: until they are summarized from
/// the callee, a call reads and writes everything, so no memory operation is
/// moved or merged across it.
#[pliron_op(
    name = "branch.call",
    format = "attr($branch_call_callee, $IdentifierAttr) ` (` operands(CharSpace(`,`)) `) : ` attr($builtin_callee_type, $TypeAttr) ` -> ` types(CharSpace(`,`))",
    attributes = (branch_call_callee: IdentifierAttr),
    verifier = "succ"
)]
#[op_traits(HasSideEffects)]
pub struct CallOp;

impl CallOp {
    /// A call to `callee`, whose type is `callee_ty`, with `args`.
    pub fn new(
        ctx: &mut Context,
        callee: Identifier,
        callee_ty: TypeHandle,
        args: Vec<Value>,
    ) -> Self {
        let results = function_results(ctx, callee_ty);
        let op = Operation::new(ctx, Self::get_concrete_op_info(), results, args, vec![], 0);
        let op = CallOp { op };
        op.set_attr_branch_call_callee(ctx, IdentifierAttr::new(callee));
        CallOpInterface::set_callee_type(&op, ctx, callee_ty);
        op
    }

    /// The function this calls.
    pub fn callee_symbol(&self, ctx: &Context) -> Identifier {
        self.get_attr_branch_call_callee(ctx)
            .expect("a call names its callee")
            .clone()
            .into()
    }
}

fn function_results(ctx: &Context, ty: TypeHandle) -> Vec<TypeHandle> {
    let ty = ty.deref(ctx);
    ty.downcast_ref::<FunctionType>()
        .expect("a callee type is a function type")
        .res_types()
        .to_vec()
}

#[op_interface_impl]
impl CallOpInterface for CallOp {
    fn callee(&self, ctx: &Context) -> CallOpCallable {
        CallOpCallable::Direct(self.callee_symbol(ctx))
    }

    fn args(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

#[op_interface_impl]
impl control_flow::CallOpInterface for CallOp {
    fn args_range(&self, ctx: &Context) -> Range<usize> {
        0..self.get_operation().deref(ctx).get_num_operands()
    }
}

#[op_interface_impl]
impl MemoryEffectsOp for CallOp {
    /// The callee's effects, as the caller sees them: an effect on memory
    /// the callee allocates itself is invisible outside it and dropped, one
    /// on a parameter applies to the argument passed for it, and one on
    /// memory at large is kept. Opaque when the callee can't be found.
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect> {
        let Some(callee) = resolve_callee(ctx, self) else {
            return vec![MemoryEffect::Opaque];
        };
        let func_op = callee.get_operation();
        let params: Vec<Value> = callee.get_entry_block(ctx).deref(ctx).arguments().collect();
        let args: Vec<Value> = self.get_operation().deref(ctx).operands().collect();
        let as_seen_by_caller = |value: Value| -> Option<Option<Value>> {
            if let Some(i) = params.iter().position(|param| *param == value) {
                return Some(Some(args[i]));
            }
            match crate::convert::defined_in(ctx, func_op, value) {
                true => None,
                false => Some(None),
            }
        };

        let mut effects = Vec::new();
        for effect in crate::interfaces::side_effects::get_nested_memory_effects(ctx, func_op) {
            let mapped = match effect {
                MemoryEffect::Read(value) => as_seen_by_caller(value)
                    .map(|it| it.map_or(MemoryEffect::Opaque, MemoryEffect::Read)),
                MemoryEffect::Write(value) => as_seen_by_caller(value)
                    .map(|it| it.map_or(MemoryEffect::Opaque, MemoryEffect::Write)),
                other => Some(other),
            };
            effects.extend(mapped);
        }
        effects
    }
}

#[op_interface_impl]
impl SymbolUserOpInterface for CallOp {
    fn verify_symbol_uses(
        &self,
        ctx: &Context,
        symbol_tables: &mut SymbolTableCollection,
    ) -> Result<()> {
        let callee_sym = self.callee_symbol(ctx);
        let Some(callee) =
            symbol_tables.lookup_symbol_in_nearest_table(ctx, self.get_operation(), &callee_sym)
        else {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::SymbolNotFound(callee_sym.to_string())
            );
        };
        let Some(func_op) = (&*callee as &dyn Op).downcast_ref::<FuncOp>() else {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::NotFunc(callee_sym.to_string())
            );
        };
        let func_ty = func_op.get_type(ctx);
        let call_ty = CallOpInterface::callee_type(self, ctx);
        if func_ty != call_ty {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::FuncTypeErr(format!(
                    "expected {}, got {}",
                    func_ty.disp(ctx),
                    call_ty.disp(ctx)
                ))
            );
        }
        Ok(())
    }

    fn used_symbols(&self, ctx: &Context) -> Vec<Identifier> {
        vec![self.callee_symbol(ctx)]
    }
}

/// The [`FuncOp`] `call` calls, looked up in the module that holds it.
pub fn resolve_callee(ctx: &Context, call: &CallOp) -> Option<FuncOp> {
    let module = call.get_operation().deref(ctx).get_parent_op(ctx)?;
    let module = enclosing_module(ctx, module);
    let symbol = call.callee_symbol(ctx);
    find_func(ctx, module, &symbol)
}

fn enclosing_module(ctx: &Context, mut op: Ptr<Operation>) -> Ptr<Operation> {
    while let Some(parent) = op.deref(ctx).get_parent_op(ctx) {
        op = parent;
    }
    op
}

/// The function named `symbol` directly inside `module`.
pub fn find_func(ctx: &Context, module: Ptr<Operation>, symbol: &Identifier) -> Option<FuncOp> {
    use pliron::{builtin::op_interfaces::SymbolOpInterface, linked_list::ContainsLinkedList};

    let region = module.deref(ctx).get_region(0);
    let block = region.deref(ctx).iter(ctx).next()?;
    block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| Operation::get_op::<FuncOp>(op, ctx))
        .find(|func| &func.get_symbol_name(ctx) == symbol)
}
