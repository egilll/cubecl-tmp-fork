use cubecl_core as cubecl;
use cubecl_core::Compiler;
use cubecl_core::prelude::*;
use cubecl_ir::{
    AddressSpace, ContextExt, GlobalState,
    dialect::{branch::RangeLoopOp, memory::LoadOp},
    types::PointerType,
};
use cubecl_wgpu::WgslCompiler;
use pliron::{
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::Op,
    operation::Operation,
    r#type::Typed,
};

#[cube(inline)]
fn aliasing_loop(values: &mut [u32]) {
    if UNIT_POS == 0 {
        let order = [values[0], values[1] + 1, values[2] * 2];
        for i in 0..3usize {
            values[ABSOLUTE_POS + i + 3] = order[(i + values[3] as usize) % 3];
        }
    }
}

fn builder() -> KernelBuilder {
    KernelBuilder::new(KernelSettings::new(
        CubeDim::new_1d(1).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

#[test]
fn a_possibly_aliasing_store_keeps_the_global_load_in_the_loop() {
    let mut builder = builder();
    let mut values = <[u32]>::expand(&BufferCompilationArg { inplace: None }, &mut builder);
    aliasing_loop::expand(&builder.scope, &mut values);
    let shader = WgslCompiler
        .compile(builder.build(), &Default::default())
        .unwrap();
    let ctx = &shader.ctx;
    let module = ctx.aux_ty::<GlobalState>().module.get_operation();
    let mut loops = Vec::new();
    walk_op(
        ctx,
        &mut loops,
        &WALKCONFIG_PREORDER_FORWARD,
        module,
        |ctx, loops, node| {
            if let IRNode::Operation(op) = node
                && Operation::get_op::<RangeLoopOp>(op, ctx).is_some()
            {
                loops.push(op);
            }
        },
    );
    assert_eq!(loops.len(), 1);
    let mut global_loads = 0;
    walk_op(
        ctx,
        &mut global_loads,
        &WALKCONFIG_PREORDER_FORWARD,
        loops[0],
        |ctx, count, node| {
            if let IRNode::Operation(op) = node
                && let Some(load) = Operation::get_op::<LoadOp>(op, ctx)
            {
                let ty = load.ptr(ctx).get_type(ctx).deref(ctx);
                if matches!(
                    ty.downcast_ref::<PointerType>().unwrap().address_space,
                    AddressSpace::Global(_)
                ) {
                    *count += 1;
                }
            }
        },
    );
    assert_eq!(
        global_loads, 1,
        "the rotation must be reloaded on every iteration"
    );
}

#[cube(inline)]
fn fixed_index_accumulators(values: &mut [u32]) {
    let mut sums = Array::<u32>::new(12usize);
    #[unroll]
    for i in 0..12usize {
        sums[i] = 0;
    }
    for j in 0..values.len() {
        #[unroll]
        for a in 0..3usize {
            #[unroll]
            for f in 0..4usize {
                sums[a * 4 + f] += values[j] * (a + f) as u32;
            }
        }
    }
    #[unroll]
    for i in 0..12usize {
        values[ABSOLUTE_POS + i] = sums[i];
    }
}

#[test]
fn constant_index_accumulators_are_split_before_target_lowering() {
    let mut builder = builder();
    let mut values = <[u32]>::expand(&BufferCompilationArg { inplace: None }, &mut builder);
    fixed_index_accumulators::expand(&builder.scope, &mut values);
    let shader = WgslCompiler
        .compile(builder.build(), &Default::default())
        .unwrap()
        .to_string();
    assert!(!shader.contains("array<u32, 12>"), "{shader}");
}
