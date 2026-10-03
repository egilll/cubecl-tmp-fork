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
    inline_for(traced, max_inline_ops, true);
}

fn inline_for(traced: &mut Traced, max_inline_ops: usize, global_pointer_params: bool) {
    InlinePass::new(InlinePolicy::Target {
        max_inline_ops,
        global_pointer_params,
    })
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
fn a_builtin_a_callee_reads_becomes_its_parameter() {
    use cubecl_ir::dialect::general::ReadBuiltinOp;

    let mut traced = trace(|scope| {
        calls_a_builtin_reader::expand(scope);
    });
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 2);
    inline_for_target(&mut traced, 2);
    assert_eq!(
        count::<CallOp>(&traced.ctx, traced.module),
        2,
        "the calls are kept"
    );

    let callee = functions(&traced)
        .into_iter()
        .find(|func| {
            cubecl_ir::attributes::EntrypointInterface::get_entrypoint_abi(func, &traced.ctx)
                .is_none()
        })
        .expect("the callee survives");
    assert_eq!(
        count::<ReadBuiltinOp>(&traced.ctx, callee.get_operation()),
        0,
        "the callee reads no builtin"
    );
    let params = callee
        .get_entry_block(&traced.ctx)
        .deref(&traced.ctx)
        .get_num_arguments();
    assert_eq!(params, 2, "`x`, then `UNIT_POS`");
}

fn functions(traced: &Traced) -> Vec<FuncOp> {
    use pliron::linked_list::ContainsLinkedList;
    let body = traced.module.deref(&traced.ctx).get_region(0);
    let block = body.deref(&traced.ctx).get_head().unwrap();
    block
        .deref(&traced.ctx)
        .iter(&traced.ctx)
        .filter_map(|op| Operation::get_op::<FuncOp>(op, &traced.ctx))
        .collect()
}

#[derive(CubeType, Clone, Copy)]
struct Curve {
    scale: f32,
    shift: f32,
    #[cube(comptime)]
    power: u32,
}

#[cube]
impl Curve {
    fn new(scale: f32, shift: f32, #[comptime] power: u32) -> Curve {
        Curve {
            scale,
            shift,
            power,
        }
    }

    #[cube(outline)]
    fn eval(&self, x: f32) -> f32 {
        let mut y = x * self.scale + self.shift;
        #[unroll]
        for _ in 0..self.power {
            y = y * x + f32::exp(y * 0.01);
        }
        y
    }
}

#[cube(outline)]
fn through_reference(curve: &Curve, x: f32) -> f32 {
    curve.eval(x) * 2.0 + curve.shift
}

#[cube]
fn method_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let a = Curve::new(x, x * 2.0, 3u32);
    let b = Curve::new(x * 3.0, x, 3u32);
    let c = Curve::new(x, x, 4u32);
    // `a` and `b` share `eval` (same comptime power), `c` has its own.
    a.eval(x) + b.eval(x + 1.0) + c.eval(x) + through_reference(&a, x) + through_reference(&b, x)
}

#[test]
fn methods_and_struct_arguments_are_outlined() {
    let traced = trace(|scope| {
        method_calls::expand(scope);
    });
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
    // Three direct `eval` calls, two `through_reference` calls, and the
    // `eval` call inside `through_reference`'s body.
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 6);
    // The kernel, `eval` for power 3, `eval` for power 4, `through_reference`.
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 4);
}

#[cube]
trait Decay: CubeType {
    fn decay(&self, x: f32) -> f32;
}

#[cube]
impl Decay for Curve {
    #[cube(outline)]
    fn decay(&self, x: f32) -> f32 {
        f32::powf(1.0 + x / (9.0 * self.scale), -1.0) * self.shift + f32::exp(-x * self.scale)
    }
}

#[cube]
fn decays<D: Decay>(law: &D, x: f32) -> f32 {
    law.decay(x) + law.decay(x * 2.0) + law.decay(x * 3.0)
}

#[cube]
fn trait_method_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let curve = Curve::new(x, x * 2.0, 3u32);
    decays::<Curve>(&curve, x)
}

#[test]
fn trait_impl_methods_are_outlined() {
    let traced = trace(|scope| {
        trait_method_calls::expand(scope);
    });
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 3);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 2);
}

#[cube(outline)]
fn weighted_sum(values: &[f32], weights: &[f32], start: usize, count: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in start..start + count {
        acc += values[i] * weights[i] + f32::exp(values[i] * 0.01);
    }
    acc
}

#[cube(outline)]
fn scale_into(values: &mut [f32], factor: f32, count: usize) {
    for i in 0..count {
        values[i] = values[i] * factor + f32::ln(factor + 1.0);
    }
}

#[cube]
fn slice_calls(values: &mut [f32], weights: &[f32]) {
    let n = UNIT_POS as usize;
    let a = weighted_sum(values, weights, n, 4);
    let b = weighted_sum(values, weights, n + 4, 4);
    scale_into(values, a + b, 8);
    scale_into(values, a - b, 8);
}

fn trace_slice_calls() -> Traced {
    trace(|scope| {
        let f32_ty = f32::__expand_as_type(scope);
        let values = scope.global(0, None, f32_ty);
        let weights = scope.global(1, None, f32_ty);
        let len = scope.const_usize(64);
        let zero = scope.const_usize(0);
        let values =
            cubecl_core::frontend::from_raw_parts::<f32>(scope, values, zero.into(), len.into());
        let weights =
            cubecl_core::frontend::from_raw_parts::<f32>(scope, weights, zero.into(), len.into());
        let mut values = values;
        slice_calls::expand(scope, &mut values, &weights);
    })
}

#[test]
fn slices_are_passed_to_a_target_that_takes_global_pointers() {
    let mut traced = trace_slice_calls();
    pliron::operation::verify_operation(traced.module, &traced.ctx).expect("the module verifies");
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 4);
    inline_for(&mut traced, 2, true);
    assert_eq!(
        count::<CallOp>(&traced.ctx, traced.module),
        4,
        "the calls are kept"
    );
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 3);
}

#[test]
fn slices_are_inlined_for_a_target_without_global_pointer_parameters() {
    let mut traced = trace_slice_calls();
    inline_for(&mut traced, 2, false);
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 0);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 1);
}

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Table {
    base: f32,
}

#[cube(outline)]
impl Table {
    // Returns a struct, which a function can't yet: traced inline.
    fn new(base: f32) -> Table {
        Table { base }
    }

    // Takes `&mut self`, which an impl-wide `outline` skips.
    fn shift(&mut self, by: f32) {
        self.base += by;
    }

    fn lookup(&self, x: f32) -> f32 {
        f32::exp(x * self.base) + f32::ln(x + self.base) * self.base
    }

    #[cube(inline)]
    fn lookup_inline(&self, x: f32) -> f32 {
        f32::exp(x * self.base) + f32::ln(x + self.base) * self.base
    }
}

#[cube]
fn table_calls() -> f32 {
    let x = f32::cast_from(UNIT_POS);
    let mut table = Table::new(x);
    table.shift(1.0);
    table.lookup(x) + table.lookup(x + 1.0) + table.lookup_inline(x) + table.lookup_inline(x)
}

#[test]
fn an_impl_marked_outline_outlines_its_methods_but_inline_ones() {
    let traced = trace(|scope| {
        table_calls::expand(scope);
    });
    assert_eq!(count::<CallOp>(&traced.ctx, traced.module), 2);
    assert_eq!(count::<FuncOp>(&traced.ctx, traced.module), 2);
}
