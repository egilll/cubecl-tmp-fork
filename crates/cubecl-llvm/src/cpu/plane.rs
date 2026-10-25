//! CPU plane operations: a plane is one unit, so every lane is lane zero,
//! a shuffle returns its own value and a ballot is the unit's predicate.

use crate::{prelude::*, shared::plane::PlaneLowering};

pub struct CpuPlane;

impl PlaneLowering for CpuPlane {
    fn lane_id(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        insert_i32_const(ctx, rw, 0)
    }

    fn shuffle(
        &self,
        _ctx: &mut Context,
        _rw: &mut DialectConversionRewriter,
        value: Value,
        _src_lane: Value,
        _value_ty: TypeHandle,
    ) -> Value {
        value
    }

    fn shuffle_xor(
        &self,
        _ctx: &mut Context,
        _rw: &mut DialectConversionRewriter,
        value: Value,
        _mask: Value,
        _value_ty: TypeHandle,
    ) -> Value {
        value
    }

    fn shuffle_up(
        &self,
        _ctx: &mut Context,
        _rw: &mut DialectConversionRewriter,
        value: Value,
        _delta: Value,
        _value_ty: TypeHandle,
    ) -> Value {
        value
    }

    fn shuffle_down(
        &self,
        _ctx: &mut Context,
        _rw: &mut DialectConversionRewriter,
        value: Value,
        _delta: Value,
        _value_ty: TypeHandle,
    ) -> Value {
        value
    }

    fn ballot_mask(
        &self,
        _ctx: &mut Context,
        _rw: &mut DialectConversionRewriter,
        predicate: Value,
    ) -> Value {
        predicate
    }

    fn elect(&self, ctx: &mut Context, rw: &mut DialectConversionRewriter) -> Value {
        insert_bool_const(ctx, rw, true)
    }
}
