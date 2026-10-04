//! A `fast_math` scope stamps the float operations it creates with its flags,
//! and nothing else: backends relax exactly those.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_ir::{
    AddressType, FastMath, Scope,
    attributes::{ATTR_FAST_MATH, FastMathAttr},
    dialect::{
        OperationPtrExt,
        math::{ExpOp, FAddOp, FMulOp},
    },
    settings::{Dim3, ExecutionMode, KernelSettings},
};
use pliron::{
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::Op,
    operation::Operation,
};

#[cube(inline)]
fn mixed(x: f32) -> f32 {
    let precise = f32::exp(x) + x;
    let relaxed = fast_math_section(x);
    precise * relaxed
}

#[cube(inline, fast_math = FastMath::ReducedPrecision | FastMath::AllowContraction)]
fn fast_math_section(x: f32) -> f32 {
    f32::exp(x) + x
}

/// (flagged, unflagged) counts of ops of type `T`.
fn flagged<T: Op>(ctx: &Context, op: Ptr<Operation>) -> (usize, usize) {
    let mut counts = (0, 0);
    walk_op(
        ctx,
        &mut counts,
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx, counts, node| {
            if let IRNode::Operation(op) = node
                && Operation::get_op::<T>(op, ctx).is_some()
            {
                match op.get_attr::<FastMathAttr>(ctx, &ATTR_FAST_MATH) {
                    Some(attr) => {
                        assert!(attr.allows(FastMath::ReducedPrecision));
                        assert!(attr.allows(FastMath::AllowContraction));
                        assert!(!attr.allows(FastMath::AllowReassociation));
                        counts.0 += 1;
                    }
                    None => counts.1 += 1,
                }
            }
        },
    );
    counts
}

#[test]
fn only_operations_under_fast_math_carry_its_flags() {
    let scope = Scope::root(KernelSettings::new(
        Dim3 { x: 1, y: 1, z: 1 },
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let x: NativeExpand<f32> = f32::__expand_cast_from(&scope, UNIT_POS::expand(&scope));
    mixed::expand(&scope, x);
    let module = scope.state().module.get_operation();
    let ctx = scope.into_context().expect("the scope owns its context");

    assert_eq!(flagged::<ExpOp>(&ctx, module), (1, 1));
    assert_eq!(flagged::<FAddOp>(&ctx, module), (1, 1));
    // The multiply is outside the section.
    assert_eq!(flagged::<FMulOp>(&ctx, module), (0, 1));
}
