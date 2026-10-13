//! Kernel implementations of the `core::ops` operators.
//!
//! Each operator has two traits. `CubeMul<Rhs>` is the kernel implementation of
//! `core::ops::Mul<Rhs>`, implemented for the same host types (`#[cube] impl Mul`
//! derives it), so any set of `Mul` impls that coexist in Rust also coexist in
//! kernels. `MulExpand<Rhs>` is what a kernel's `a * b` calls on expand values: it
//! finds the host types behind both operands and calls their `CubeMul`, with the
//! output type `Mul` names.

use core::ops::*;

use crate::frontend::{CubePrimitive, CubeType, ExpandTypeClone, NativeExpand};
use crate::ir::Scope;

use super::*;

/// A kernel value as an operator operand: the expand type of `Host`, or a
/// reference to one, which the operator reads.
pub trait Operand {
    type Host: CubeType + ?Sized;
    fn __expand_operand(self, scope: &Scope) -> <Self::Host as CubeType>::ExpandType;
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> Operand for NativeExpand<T> {
    type Host = T;
    fn __expand_operand(self, _: &Scope) -> NativeExpand<T> {
        self
    }
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> Operand for &NativeExpand<T> {
    type Host = T;
    fn __expand_operand(self, _: &Scope) -> NativeExpand<T> {
        self.clone_unchecked()
    }
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> Operand for &mut NativeExpand<T> {
    type Host = T;
    fn __expand_operand(self, _: &Scope) -> NativeExpand<T> {
        (*self).clone_unchecked()
    }
}

/// A kernel value a compound assignment writes: the expand type of `Host`,
/// or a mutable reference to one.
pub trait OperandPlace: Operand {
    fn __expand_place(&mut self) -> &mut <Self::Host as CubeType>::ExpandType;
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> OperandPlace for NativeExpand<T> {
    fn __expand_place(&mut self) -> &mut NativeExpand<T> {
        self
    }
}

impl<T: CubeType<ExpandType = NativeExpand<T>>> OperandPlace for &mut NativeExpand<T> {
    fn __expand_place(&mut self) -> &mut NativeExpand<T> {
        self
    }
}

macro_rules! binary_operator {
    ($trait: ident, $method: ident) => {
        paste::paste! {
            #[doc = concat!("The kernel implementation of `core::ops::", stringify!($trait), "<Rhs>`, for the same types.")]
            pub trait [<Cube $trait>]<Rhs: CubeType = Self>: CubeType<ExpandType: Operand<Host = Self>> + Sized + $trait<Rhs, Output: CubeType> {
                fn [<__expand_ $method>](
                    scope: &Scope,
                    lhs: Self::ExpandType,
                    rhs: Rhs::ExpandType,
                ) -> <Self::Output as CubeType>::ExpandType;
            }

            #[doc = concat!("`", stringify!($trait), "` on kernel values, through their host types' [`Cube", stringify!($trait), "`].")]
            pub trait [<$trait Expand>]<Rhs> {
                type Output;
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: Rhs) -> Self::Output;
            }

            impl<L: Operand, R: Operand<Host: Sized>> [<$trait Expand>]<R> for L
            where
                L::Host: [<Cube $trait>]<R::Host>,
            {
                type Output = <<L::Host as $trait<R::Host>>::Output as CubeType>::ExpandType;
                fn [<__expand_ $method _method>](self, scope: &Scope, rhs: R) -> Self::Output {
                    let lhs = self.__expand_operand(scope);
                    let rhs = rhs.__expand_operand(scope);
                    <L::Host as [<Cube $trait>]<R::Host>>::[<__expand_ $method>](scope, lhs, rhs)
                }
            }

            impl<T> [<Cube $trait>] for T
            where
                T: $trait<Output = T> + CubePrimitive<Scalar: [<$trait NativeExpand>]>,
            {
                fn [<__expand_ $method>](
                    scope: &Scope,
                    lhs: NativeExpand<T>,
                    rhs: NativeExpand<T>,
                ) -> NativeExpand<T> {
                    T::Scalar::[<__expand_native_ $method>](scope, lhs.into(), rhs.into()).into()
                }
            }

            /// A scalar with the native operator.
            pub trait [<Scalar $trait>]: [<Cube $trait>] + [<$trait NativeExpand>] + $trait<Output = Self> {}
            impl<T: [<Cube $trait>] + [<$trait NativeExpand>] + $trait<Output = Self>> [<Scalar $trait>] for T {}
        }
    };
}

macro_rules! assign_operator {
    ($trait: ident, $base_trait: ident, $method: ident, $base_method: ident) => {
        paste::paste! {
            #[doc = concat!("The kernel implementation of `core::ops::", stringify!($trait), "<Rhs>`, for the same types.")]
            pub trait [<Cube $trait>]<Rhs: CubeType = Self>: CubeType<ExpandType: OperandPlace<Host = Self>> + Sized + $trait<Rhs> {
                fn [<__expand_ $method>](scope: &Scope, lhs: &mut Self::ExpandType, rhs: Rhs::ExpandType);
            }

            #[doc = concat!("`", stringify!($trait), "` on kernel values, through their host types' [`Cube", stringify!($trait), "`].")]
            pub trait [<$trait Expand>]<Rhs> {
                fn [<__expand_ $method _method>](&mut self, scope: &Scope, rhs: Rhs);
            }

            impl<L: OperandPlace, R: Operand<Host: Sized>> [<$trait Expand>]<R> for L
            where
                L::Host: [<Cube $trait>]<R::Host>,
            {
                fn [<__expand_ $method _method>](&mut self, scope: &Scope, rhs: R) {
                    let rhs = rhs.__expand_operand(scope);
                    <L::Host as [<Cube $trait>]<R::Host>>::[<__expand_ $method>](
                        scope,
                        self.__expand_place(),
                        rhs,
                    )
                }
            }

            impl<T> [<Cube $trait>] for T
            where
                T: $trait + CubePrimitive<Scalar: [<$base_trait NativeExpand>]>,
            {
                fn [<__expand_ $method>](scope: &Scope, lhs: &mut NativeExpand<T>, rhs: NativeExpand<T>) {
                    super::base::assign_binop_expand(
                        scope,
                        lhs,
                        rhs,
                        T::Scalar::[<__expand_native_ $base_method>],
                    );
                }
            }
        }
    };
}

macro_rules! unary_operator {
    ($trait: ident, $method: ident, $native: expr) => {
        paste::paste! {
            #[doc = concat!("The kernel implementation of `core::ops::", stringify!($trait), "`, for the same types.")]
            pub trait [<Cube $trait>]: CubeType<ExpandType: Operand<Host = Self>> + $trait<Output: CubeType> {
                fn [<__expand_ $method>](
                    scope: &Scope,
                    input: Self::ExpandType,
                ) -> <Self::Output as CubeType>::ExpandType;
            }

            #[doc = concat!("`", stringify!($trait), "` on kernel values, through their host type's [`Cube", stringify!($trait), "`].")]
            pub trait [<$trait Expand>] {
                type Output;
                fn [<__expand_ $method _method>](self, scope: &Scope) -> Self::Output;
            }

            impl<X: Operand> [<$trait Expand>] for X
            where
                X::Host: [<Cube $trait>],
            {
                type Output = <<X::Host as $trait>::Output as CubeType>::ExpandType;
                fn [<__expand_ $method _method>](self, scope: &Scope) -> Self::Output {
                    let input = self.__expand_operand(scope);
                    <X::Host as [<Cube $trait>]>::[<__expand_ $method>](scope, input)
                }
            }

            impl<T> [<Cube $trait>] for T
            where
                T: $trait<Output = T> + CubePrimitive<Scalar: [<$trait NativeExpand>]>,
            {
                fn [<__expand_ $method>](scope: &Scope, input: NativeExpand<T>) -> NativeExpand<T> {
                    $native(scope, input)
                }
            }

            /// A scalar with the native operator.
            pub trait [<Scalar $trait>]: [<Cube $trait>] + [<$trait NativeExpand>] + $trait<Output = Self> {}
            impl<T: [<Cube $trait>] + [<$trait NativeExpand>] + $trait<Output = Self>> [<Scalar $trait>] for T {}
        }
    };
}

binary_operator!(Add, add);
binary_operator!(Sub, sub);
binary_operator!(Mul, mul);
binary_operator!(Div, div);
binary_operator!(Rem, rem);
binary_operator!(BitAnd, bitand);
binary_operator!(BitOr, bitor);
binary_operator!(BitXor, bitxor);
binary_operator!(Shl, shl);
binary_operator!(Shr, shr);

assign_operator!(AddAssign, Add, add_assign, add);
assign_operator!(SubAssign, Sub, sub_assign, sub);
assign_operator!(MulAssign, Mul, mul_assign, mul);
assign_operator!(DivAssign, Div, div_assign, div);
assign_operator!(RemAssign, Rem, rem_assign, rem);
assign_operator!(BitAndAssign, BitAnd, bitand_assign, bitand);
assign_operator!(BitOrAssign, BitOr, bitor_assign, bitor);
assign_operator!(BitXorAssign, BitXor, bitxor_assign, bitxor);
assign_operator!(ShlAssign, Shl, shl_assign, shl);
assign_operator!(ShrAssign, Shr, shr_assign, shr);

unary_operator!(Neg, neg, |scope, input: NativeExpand<T>| {
    T::Scalar::__expand_native_neg(scope, input.expand).into()
});
unary_operator!(Not, not, not::expand::<T>);
