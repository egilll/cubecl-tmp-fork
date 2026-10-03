//! `#[cube(outline)]` traces a function once per specialization into a
//! function of the kernel module, and its calls call it.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_ir::{
    AddressType, Scope,
    dialect::call::CallOp,
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use cubecl_opt::passes::inline::{InlinePass, InlinePolicy};
use pliron::{
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::Op,
    operation::Operation,
    pass::{AnalysisManager, Pass},
};

#[cube(outline)]
fn heavy(x: f32, y: f32) -> f32 {
    f32::exp(x) * y + f32::powf(x, y) - f32::ln(y + 2.0)
}

#[cube]
fn heavy_inlined(x: f32, y: f32) -> f32 {
    f32::exp(x) * y + f32::powf(x, y) - f32::ln(y + 2.0)
}

#[cube]
fn four_outlined_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let mut acc = x;
    #[unroll]
    for _ in 0..4u32 {
        acc = heavy(x, acc);
    }
    acc
}

#[cube]
fn sixteen_inlined_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let mut acc = x;
    #[unroll]
    for _ in 0..16u32 {
        acc = heavy_inlined(x, acc);
    }
    acc
}

#[cube]
fn sixteen_outlined_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let mut acc = x;
    #[unroll]
    for _ in 0..16u32 {
        acc = heavy(x, acc);
    }
    acc
}

#[cube]
fn a_constant_argument_specializes() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    heavy(x, x) + heavy(x, 2.0) + heavy(x, 2.0) + heavy(x, 3.0)
}

#[cube(outline)]
fn scaled<F: Float>(x: F, #[comptime] factor: u32) -> F {
    x * F::cast_from(factor)
}

#[cube]
fn generics_and_comptime_specialize() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let a = scaled::<f32>(x, 2u32);
    let b = scaled::<f32>(x, 2u32);
    let c = scaled::<f32>(x, 3u32);
    let d = f32::cast_from(scaled::<f64>(f64::cast_from(x), 2u32));
    a + b + c + d
}

#[cube(outline)]
fn stops_the_kernel(x: f32) {
    if x > 1.0 {
        terminate!();
    }
}

#[cube]
fn terminating_callee() {
    let x = f32::cast_from(UNIT_POS);
    stops_the_kernel(x);
    stops_the_kernel(x);
}

fn kernel() -> Scope {
    Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ))
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

struct Traced {
    module: Ptr<Operation>,
    ctx: Context,
    ops: usize,
}

fn trace(body: impl FnOnce(&Scope)) -> Traced {
    let scope = kernel();
    body(&scope);
    let ops = scope.op_count();
    let module = scope.state().module.get_operation();
    let ctx = scope.into_context().expect("the scope owns its context");
    Traced { module, ctx, ops }
}

fn inline_all(traced: &mut Traced) {
    InlinePass::new(InlinePolicy::All)
        .run(
            traced.module,
            &mut traced.ctx,
            &mut AnalysisManager::default(),
        )
        .expect("inlining succeeds");
    pliron::operation::verify_operation(traced.module, &traced.ctx)
        .expect("the inlined module verifies");
}

#[test]
fn calls_with_the_same_specialization_share_one_function() {
    let mut traced = trace(|scope| {
        four_outlined_calls::expand(scope);
    });
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 4);
    assert_eq!(
        count::<FuncOp>(&traced.ctx, traced.module),
        2,
        "the kernel and `heavy`"
    );

    inline_all(&mut traced);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}

#[test]
fn an_outlined_function_is_traced_once() {
    let outlined = trace(|scope| {
        sixteen_outlined_calls::expand(scope);
    });
    let inlined = trace(|scope| {
        sixteen_inlined_calls::expand(scope);
    });
    // One body and sixteen calls against sixteen bodies.
    assert!(
        outlined.ops * 2 < inlined.ops,
        "outlined {} ops, inlined {} ops",
        outlined.ops,
        inlined.ops
    );
}

#[test]
fn a_constant_argument_gets_its_own_function() {
    let traced = trace(|scope| {
        a_constant_argument_specializes::expand(scope);
    });
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 4);
    // `heavy(x, x)`, `heavy(x, 2.0)` twice, and `heavy(x, 3.0)`.
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1 + 3);
}

#[test]
fn generic_types_and_comptime_arguments_specialize() {
    let traced = trace(|scope| {
        generics_and_comptime_specialize::expand(scope);
    });
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 4);
    // `scaled::<f32>(_, 2)` twice, `scaled::<f32>(_, 3)` and `scaled::<f64>(_, 2)`.
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1 + 3);
}

#[test]
fn a_body_that_terminates_the_kernel_is_traced_inline() {
    let traced = trace(|scope| {
        terminating_callee::expand(scope);
    });
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}

#[cube(outline)]
fn reads_a_builtin(x: f32) -> f32 {
    x + f32::cast_from(UNIT_POS) * 2.0 + f32::exp(x) * f32::ln(x + 1.0)
}

#[cube]
fn calls_a_builtin_reader() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    reads_a_builtin(x) + reads_a_builtin(x + 1.0)
}

#[cube]
fn one_call() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    heavy(x, x)
}

fn inline_for_target(traced: &mut Traced, max_inline_ops: usize) {
    InlinePass::new(InlinePolicy::Target { max_inline_ops })
        .run(
            traced.module,
            &mut traced.ctx,
            &mut AnalysisManager::default(),
        )
        .expect("inlining succeeds");
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
}

#[test]
fn a_target_keeps_calls_to_a_large_callee() {
    let mut traced = trace(|scope| {
        four_outlined_calls::expand(scope);
    });
    inline_for_target(&mut traced, 2);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 4);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 2);

    // The kernel comes after the function it calls.
    use pliron::linked_list::ContainsLinkedList;
    let body = traced.module.deref(&traced.ctx).get_region(0);
    let block = body.deref(&traced.ctx).get_head().unwrap();
    let last = block.deref(&traced.ctx).get_tail().unwrap();
    let last = Operation::get_op::<FuncOp>(last, &traced.ctx).expect("a function");
    assert!(
        cubecl_ir::attributes::EntrypointInterface::get_entrypoint_abi(&last, &traced.ctx)
            .is_some()
    );
}

#[test]
fn a_target_inlines_a_small_callee() {
    let mut traced = trace(|scope| {
        four_outlined_calls::expand(scope);
    });
    inline_for_target(&mut traced, 10_000);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}

#[test]
fn a_target_inlines_a_callee_with_one_call_site() {
    let mut traced = trace(|scope| {
        one_call::expand(scope);
    });
    inline_for_target(&mut traced, 2);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}

#[test]
fn a_target_inlines_a_callee_that_reads_kernel_state() {
    let mut traced = trace(|scope| {
        calls_a_builtin_reader::expand(scope);
    });
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 2);
    inline_for_target(&mut traced, 2);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}
