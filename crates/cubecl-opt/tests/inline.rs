//! Inlining replaces a call with its callee's body, wires the callee's
//! arguments and results to the call's, and deletes the functions it leaves
//! without users.

use cubecl_ir::{
    AddressType, OpInserter, Scope,
    dialect::{
        branch::ReturnOp,
        call::CallOp,
        math::IAddOp,
        memory::{IndexOp, StoreOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
    types::scalar::IndexType,
};
use cubecl_opt::passes::inline::{InlinePass, InlinePolicy};
use pliron::{
    builtin::{op_interfaces::SymbolOpInterface, ops::FuncOp, types::FunctionType},
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, Pass},
    r#type::Typed,
    value::Value,
};

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

/// `fn twice(x) { x + x }`, as a private function of the kernel's module.
fn define_twice(scope: &Scope) -> FuncOp {
    let index = IndexType::get(scope.ctx_mut()).to_handle();
    let ty = FunctionType::get(scope.ctx(), vec![index], vec![index]);
    let name = scope.func_ident(Some("twice"));
    let func = FuncOp::new(scope.ctx_mut(), name, ty);
    let body = func.get_entry_block(scope.ctx());
    let x = body.deref(scope.ctx()).get_argument(0);

    let child = scope.func_child(OpInserter::new_at_block_end(body));
    let sum = child.register_with_result(&IAddOp::new(scope.ctx_mut(), x, x));
    child.register(&ReturnOp::new_with_value(scope.ctx_mut(), sum));
    scope.register_func(func);
    func
}

fn call(scope: &Scope, func: FuncOp, arg: Value) -> Value {
    let ty = func.get_type(scope.ctx());
    let name = func.get_symbol_name(scope.ctx());
    let call = CallOp::new(scope.ctx_mut(), name, ty, vec![arg]);
    scope.register(&call);
    call.get_operation().deref(scope.ctx()).get_result(0)
}

fn store_out(scope: &Scope, value: Value) {
    let index_ty = IndexType::get(scope.ctx_mut()).to_handle();
    let out = scope.global(0, None, index_ty);
    let zero = scope.const_usize(0);
    let ptr = scope.register_with_result(&IndexOp::new(scope.ctx_mut(), out, zero, None));
    scope.register(&StoreOp::new(scope.ctx_mut(), ptr, value));
}

fn count<T: Op>(ctx: &Context, op: Ptr<Operation>) -> usize {
    let mut n = 0;
    walk_op(
        ctx,
        &mut n,
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx, n, node| {
            if let IRNode::Operation(op) = node
                && Operation::get_op::<T>(op, ctx).is_some()
            {
                *n += 1;
            }
        },
    );
    n
}

fn inline_all(scope: Scope) -> (Ptr<Operation>, Context) {
    let module = scope.state().module.get_operation();
    let mut ctx = scope.into_context().expect("the scope owns its context");
    InlinePass::new(InlinePolicy::All)
        .run(module, &mut ctx, &mut AnalysisManager::default())
        .expect("inlining succeeds");
    pliron::operation::verify_operation(module, &ctx).expect("the inlined module verifies");
    (module, ctx)
}

#[test]
fn a_call_becomes_its_callees_body_and_the_callee_is_deleted() {
    let scope = kernel();
    let twice = define_twice(&scope);
    let two = scope.const_usize(2);
    let result = call(&scope, twice, two);
    store_out(&scope, result);

    let (module, ctx) = inline_all(scope);
    assert_eq!(count::<CallOp>(&ctx, module), 0);
    assert_eq!(count::<FuncOp>(&ctx, module), 1, "only the kernel is left");
    assert_eq!(count::<IAddOp>(&ctx, module), 1);
    assert_eq!(
        count::<ReturnOp>(&ctx, module),
        1,
        "only the kernel's return"
    );
}

#[test]
fn every_call_site_gets_its_own_copy() {
    let scope = kernel();
    let twice = define_twice(&scope);
    let two = scope.const_usize(2);
    let four = call(&scope, twice, two);
    let eight = call(&scope, twice, four);
    store_out(&scope, eight);

    let (module, ctx) = inline_all(scope);
    assert_eq!(count::<CallOp>(&ctx, module), 0);
    assert_eq!(count::<IAddOp>(&ctx, module), 2);

    // The second copy adds the first copy's result to itself.
    let mut adds = Vec::new();
    walk_op(
        &ctx,
        &mut adds,
        &WALKCONFIG_PREORDER_FORWARD,
        module,
        |ctx, adds, node| {
            if let IRNode::Operation(op) = node
                && let Some(add) = Operation::get_op::<IAddOp>(op, ctx)
            {
                adds.push(add);
            }
        },
    );
    let first = adds[0].get_operation().deref(&ctx).get_result(0);
    let second = adds[1].get_operation();
    let operands: Vec<Value> = second.deref(&ctx).operands().collect();
    assert_eq!(operands, vec![first, first]);
    assert_eq!(first.get_type(&ctx), operands[0].get_type(&ctx));
}

#[test]
fn calls_inside_a_callee_are_inlined_too() {
    let scope = kernel();
    let twice = define_twice(&scope);

    // `fn quadruple(x) { twice(twice(x)) }`
    let index = IndexType::get(scope.ctx_mut()).to_handle();
    let ty = FunctionType::get(scope.ctx(), vec![index], vec![index]);
    let name = scope.func_ident(Some("quadruple"));
    let quadruple = FuncOp::new(scope.ctx_mut(), name, ty);
    let body = quadruple.get_entry_block(scope.ctx());
    let x = body.deref(scope.ctx()).get_argument(0);
    let child = scope.func_child(OpInserter::new_at_block_end(body));
    let once = call(&child, twice, x);
    let again = call(&child, twice, once);
    child.register(&ReturnOp::new_with_value(scope.ctx_mut(), again));
    drop(child);
    scope.register_func(quadruple);

    let one = scope.const_usize(1);
    let result = call(&scope, quadruple, one);
    store_out(&scope, result);

    let (module, ctx) = inline_all(scope);
    assert_eq!(count::<CallOp>(&ctx, module), 0);
    assert_eq!(count::<FuncOp>(&ctx, module), 1);
    assert_eq!(count::<IAddOp>(&ctx, module), 2);
}
