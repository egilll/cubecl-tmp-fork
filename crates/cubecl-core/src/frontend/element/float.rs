use cubecl_ir::{
    ConstantValue, ElemType, Scope,
    types::scalar::{BFloat16Type, Float16Type, Float32Type, Float64Type},
};
use half::{bf16, f16};
use pliron::r#type::TypeHandle;

use crate::{self as cubecl, ir::FloatKind, prelude::*};

use super::Numeric;

mod fp4;
mod fp6;
mod fp8;
mod relaxed;
mod tensor_float;

/// Refuses `f64` in a kernel traced for a device with no use of it at all,
/// once per kernel. Without the refusal the backend's own compiler fails
/// later, far from the cause, which is most often an untyped float literal
/// that nothing typed met, so that Rust settled it as `f64`.
fn refuse_unsupported(scope: &Scope, kind: FloatKind) {
    if kind != FloatKind::F64 {
        return;
    }
    let ty = ElemType::Float(kind);
    let Some(properties) = scope.state().device_properties.clone() else {
        return;
    };
    if !properties.type_usage(ty).is_empty() || scope.state().unsupported_types.contains(&ty) {
        return;
    }
    scope.state_mut().unsupported_types.push(ty);
    scope.push_error(
        "this device has no f64, but the kernel uses it: an untyped float literal that \
         meets nothing typed, such as `let mut sum = 0.0;` read only by literals, settles as \
         f64; give it a type (`0.0f32`) or combine it with a typed value",
    );
}

/// Floating point numbers. Used as input in float kernels
pub trait Float:
    Numeric
    + FloatOps
    + ScalarNeg
    + ScalarExp
    + ScalarLog
    + ScalarLog1p
    + ScalarExpm1
    + ScalarCos
    + ScalarSin
    + ScalarTan
    + ScalarTanh
    + ScalarSinh
    + ScalarCosh
    + ScalarArcCos
    + ScalarArcSin
    + ScalarArcTan
    + ScalarArcSinh
    + ScalarArcCosh
    + ScalarArcTanh
    + ScalarDegrees
    + ScalarRadians
    + ScalarArcTan2
    + ScalarPowf
    + Powi<i32>
    + ScalarHypot
    + ScalarRhypot
    + ScalarSqrt
    + ScalarInverseSqrt
    + ScalarRound
    + ScalarFloor
    + ScalarCeil
    + ScalarTrunc
    + ScalarErf
    + ScalarRecip
    + ScalarMagnitude
    + Normalize
    + ScalarDot
    + IsNan
    + IsInf
    + Into<Self::ExpandType>
    + core::ops::Neg<Output = Self>
    + core::cmp::PartialOrd
    + core::cmp::PartialEq
{
    const DIGITS: u32;
    const EPSILON: Self;
    const INFINITY: Self;
    const MANTISSA_DIGITS: u32;
    const MAX_10_EXP: i32;
    const MAX_EXP: i32;
    const MIN_10_EXP: i32;
    const MIN_EXP: i32;
    const MIN_POSITIVE: Self;
    const NAN: Self;
    const NEG_INFINITY: Self;
    const RADIX: u32;

    fn new(val: f32) -> Self;
    fn __expand_new(scope: &Scope, val: f32) -> <Self as CubeType>::ExpandType {
        __expand_new(scope, val)
    }
}

#[cube]
pub trait FloatOps: CubePartialOrd + Sized {
    fn min(self, other: Self) -> Self {
        cubecl::prelude::min(self, other)
    }

    fn max(self, other: Self) -> Self {
        cubecl::prelude::max(self, other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        clamp(self, min, max)
    }
}

impl<T: Float> FloatOps for T {}
impl<T: FloatOps + CubePrimitive> FloatOpsExpand for NativeExpand<T> {
    fn __expand_min_method(self, scope: &Scope, other: Self) -> Self {
        min::expand(scope, self, other)
    }

    fn __expand_max_method(self, scope: &Scope, other: Self) -> Self {
        max::expand(scope, self, other)
    }

    fn __expand_clamp_method(self, scope: &Scope, min: Self, max: Self) -> Self {
        clamp::expand(scope, self, min, max)
    }
}

macro_rules! impl_float {
    (half $primitive:ident, $ty: ty, $kind:ident) => {
        impl_float!($primitive, $ty, $kind, |val| $primitive::from_f64(val));
    };
    ($primitive:ident, $ty: ty, $kind:ident) => {
        impl_float!($primitive, $ty, $kind, |val| val as $primitive);
    };
    ($primitive:ident, $ty: ty, $kind:ident, $new:expr) => {
        impl CubeType for $primitive {
            type ExpandType = NativeExpand<$primitive>;
        }

        impl CubeDebug for $primitive {}
        impl crate::frontend::call::CallArg for $primitive {}
        impl Scalar for $primitive {
            fn elem_type(scope: &Scope) -> ElemType {
                refuse_unsupported(scope, FloatKind::$kind);
                FloatKind::$kind.into()
            }
            fn elem_type_native() -> ElemType {
                FloatKind::$kind.into()
            }
        }
        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = Const<1>;
            type WithScalar<S: Scalar> = S;

            /// Return the element type to use on GPU
            fn __expand_as_type(scope: &Scope) -> TypeHandle {
                refuse_unsupported(scope, FloatKind::$kind);
                <$ty>::get(scope.ctx()).into()
            }

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::Float(value) = value else {
                    unreachable!()
                };
                $new(value)
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl IntoExpand for $primitive {
            type Expand = NativeExpand<$primitive>;

            fn into_expand(self, _: &Scope) -> Self::Expand {
                self.into()
            }
        }

        impl Numeric for $primitive {
            fn min_value() -> Self {
                <Self as num_traits::Float>::min_value()
            }
            fn max_value() -> Self {
                <Self as num_traits::Float>::max_value()
            }
        }

        impl NativeAssign for $primitive {}

        impl IntoMut for $primitive {
            fn into_mut(self, _scope: &Scope) -> Self {
                self
            }
        }

        impl Float for $primitive {
            const DIGITS: u32 = $primitive::DIGITS;
            const EPSILON: Self = $primitive::EPSILON;
            const INFINITY: Self = $primitive::INFINITY;
            const MANTISSA_DIGITS: u32 = $primitive::MANTISSA_DIGITS;
            const MAX_10_EXP: i32 = $primitive::MAX_10_EXP;
            const MAX_EXP: i32 = $primitive::MAX_EXP;
            const MIN_10_EXP: i32 = $primitive::MIN_10_EXP;
            const MIN_EXP: i32 = $primitive::MIN_EXP;
            const MIN_POSITIVE: Self = $primitive::MIN_POSITIVE;
            const NAN: Self = $primitive::NAN;
            const NEG_INFINITY: Self = $primitive::NEG_INFINITY;
            const RADIX: u32 = $primitive::RADIX;

            fn new(val: f32) -> Self {
                $new(val as f64)
            }
        }

        impl_scalar_launch!($primitive);
    };
}

impl_float!(half f16, Float16Type, F16);
impl_float!(half bf16,  BFloat16Type, BF16);
impl_float!(f32, Float32Type, F32);
impl_float!(f64, Float64Type, F64);
