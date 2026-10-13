//! How a kernel passes arguments to device functions and constructors.
//!
//! A call site converts each argument with `Into`. A parameter whose type is
//! a generic `T` takes an [`ExpandArg<T>`] instead of `T`'s expand type, which
//! Rust can't see through to `T`. An expand value names its host
//! ([`Expanded`]) and a [`LiteralArg`] keeps its own type, so either determines
//! `T` as the argument would in Rust: `scale(x)` needs no turbofish, and an
//! untyped `1.0` stays a float literal until something else settles it.

use cubecl_common::{e2m1, e2m1x2, e2m3, e3m2, e4m3, e5m2, flex32, tf32, ue8m0};
use half::{bf16, f16};

use crate::frontend::{CubeType, Expanded, NativeExpand};

/// An argument for a parameter of host type `T`.
pub struct ExpandArg<T: CubeType + ?Sized>(T::ExpandType);

impl<T: CubeType + ?Sized> ExpandArg<T> {
    pub fn into_inner(self) -> T::ExpandType {
        self.0
    }
}

impl<E: Expanded> From<E> for ExpandArg<E::Host> {
    fn from(value: E) -> Self {
        Self(value)
    }
}

/// An untyped numeric literal passed as an argument, which converts to the
/// parameter's type: a comptime value, an expand value, or an [`ExpandArg`]
/// whose type it decides.
pub struct LiteralArg<T>(pub T);

impl<T> From<LiteralArg<T>> for NativeExpand<T>
where
    NativeExpand<T>: From<T>,
{
    fn from(literal: LiteralArg<T>) -> Self {
        literal.0.into()
    }
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> From<LiteralArg<T>> for ExpandArg<T>
where
    NativeExpand<T>: From<T>,
{
    fn from(literal: LiteralArg<T>) -> Self {
        Self(literal.0.into())
    }
}

macro_rules! literal {
    ($($ty:ty),*) => {$(
        impl From<LiteralArg<$ty>> for $ty {
            fn from(literal: LiteralArg<$ty>) -> Self {
                literal.0
            }
        }
    )*};
}

literal!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64
);

macro_rules! constant {
    ($($ty:ty),*) => {$(
        impl From<$ty> for ExpandArg<$ty> {
            fn from(value: $ty) -> Self {
                Self(value.into())
            }
        }
    )*};
}

constant!(
    u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f16, bf16, flex32, tf32, f32, f64, e2m1,
    e2m1x2, e2m3, e3m2, e4m3, e5m2, ue8m0, bool
);
