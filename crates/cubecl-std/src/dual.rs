//! Forward-mode dual numbers: a value and `N` partial derivatives, carried
//! through arithmetic and elementary functions.
//!
//! Write a formula once over [`Real`]; instantiated with `f32` it computes
//! the value, instantiated with [`Dual<N>`] it also computes the derivatives
//! with respect to `N` chosen inputs in the same pass. That suits problems
//! with few parameters and many terms, and leaves no hand-written gradient
//! to drift from the formula.

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// A real number a formula can be generic over: `f32`, or [`Dual`].
///
/// Inside a `#[cube]` function generic over `R: Real`, use the operators and
/// the path form of the functions (`R::exp(x)`, `R::constant(0.5)`). The
/// operator bounds sit on the expand type, so `a + b` traces into the right
/// arithmetic for either.
pub trait Real:
    CubeType<
        ExpandType: AddExpand<Output = <Self as CubeType>::ExpandType>
                        + SubExpand<Output = <Self as CubeType>::ExpandType>
                        + MulExpand<Output = <Self as CubeType>::ExpandType>
                        + DivExpand<Output = <Self as CubeType>::ExpandType>
                        + NegExpand
                        + Copy,
    > + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + Copy
    + Send
    + Sync
    + 'static
{
    /// A constant (zero derivatives).
    fn constant(x: f32) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }
    fn exp(x: Self) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }
    fn ln(x: Self) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }
    fn sin(x: Self) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }
    fn cos(x: Self) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }
    fn sqrt(x: Self) -> Self {
        let _ = x;
        cubecl::unexpanded!()
    }

    fn __expand_constant(scope: &Scope, x: NativeExpand<f32>) -> Self::ExpandType;
    fn __expand_exp(scope: &Scope, x: Self::ExpandType) -> Self::ExpandType;
    fn __expand_ln(scope: &Scope, x: Self::ExpandType) -> Self::ExpandType;
    fn __expand_sin(scope: &Scope, x: Self::ExpandType) -> Self::ExpandType;
    fn __expand_cos(scope: &Scope, x: Self::ExpandType) -> Self::ExpandType;
    fn __expand_sqrt(scope: &Scope, x: Self::ExpandType) -> Self::ExpandType;
}

impl Real for f32 {
    fn __expand_constant(_scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        x
    }
    fn __expand_exp(scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        <f32 as Exp>::__expand_exp(scope, x)
    }
    fn __expand_ln(scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        <f32 as Log>::__expand_ln(scope, x)
    }
    fn __expand_sin(scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        <f32 as Sin>::__expand_sin(scope, x)
    }
    fn __expand_cos(scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        <f32 as Cos>::__expand_cos(scope, x)
    }
    fn __expand_sqrt(scope: &Scope, x: NativeExpand<f32>) -> NativeExpand<f32> {
        <f32 as Sqrt>::__expand_sqrt(scope, x)
    }
}

/// A value and its derivatives with respect to `N` inputs.
#[derive(CubeType, Clone, Copy, Debug)]
#[expand(derive(Clone, Copy))]
pub struct Dual<N: Size> {
    pub value: f32,
    pub tangent: Vector<f32, N>,
}

#[cube]
impl<N: Size> Dual<N> {
    /// The `index`-th input: derivative one with respect to itself.
    pub fn variable(value: f32, #[comptime] index: usize) -> Dual<N> {
        let mut tangent = Vector::<f32, N>::new(0.0);
        tangent.insert(index, 1.0);
        Dual::<N> { value, tangent }
    }

    /// Scale by the derivative of an elementary function at the value.
    fn chain(&self, value: f32, derivative: f32) -> Dual<N> {
        Dual::<N> {
            value,
            tangent: self.tangent * Vector::new(derivative),
        }
    }
}

#[cube]
impl<N: Size> core::ops::Add for Dual<N> {
    type Output = Dual<N>;
    fn add(self, rhs: Dual<N>) -> Dual<N> {
        Dual::<N> {
            value: self.value + rhs.value,
            tangent: self.tangent + rhs.tangent,
        }
    }
}

#[cube]
impl<N: Size> core::ops::Sub for Dual<N> {
    type Output = Dual<N>;
    fn sub(self, rhs: Dual<N>) -> Dual<N> {
        Dual::<N> {
            value: self.value - rhs.value,
            tangent: self.tangent - rhs.tangent,
        }
    }
}

#[cube]
impl<N: Size> core::ops::Mul for Dual<N> {
    type Output = Dual<N>;
    fn mul(self, rhs: Dual<N>) -> Dual<N> {
        Dual::<N> {
            value: self.value * rhs.value,
            tangent: self.tangent * Vector::new(rhs.value) + rhs.tangent * Vector::new(self.value),
        }
    }
}

#[cube]
impl<N: Size> core::ops::Div for Dual<N> {
    type Output = Dual<N>;
    fn div(self, rhs: Dual<N>) -> Dual<N> {
        // The value divides as f32 does, so a formula's value is the same
        // whether it runs on f32 or on duals.
        let value = self.value / rhs.value;
        Dual::<N> {
            value,
            tangent: (self.tangent - rhs.tangent * Vector::new(value)) / Vector::new(rhs.value),
        }
    }
}

#[cube]
impl<N: Size> core::ops::Neg for Dual<N> {
    type Output = Dual<N>;
    fn neg(self) -> Dual<N> {
        Dual::<N> {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}

#[cube]
fn dual_constant<N: Size>(x: f32) -> Dual<N> {
    Dual::<N> {
        value: x,
        tangent: Vector::new(0.0),
    }
}

#[cube]
fn dual_exp<N: Size>(x: Dual<N>) -> Dual<N> {
    let e = x.value.exp();
    x.chain(e, e)
}

#[cube]
fn dual_ln<N: Size>(x: Dual<N>) -> Dual<N> {
    x.chain(x.value.ln(), 1.0 / x.value)
}

#[cube]
fn dual_sin<N: Size>(x: Dual<N>) -> Dual<N> {
    x.chain(x.value.sin(), x.value.cos())
}

#[cube]
fn dual_cos<N: Size>(x: Dual<N>) -> Dual<N> {
    x.chain(x.value.cos(), -x.value.sin())
}

#[cube]
fn dual_sqrt<N: Size>(x: Dual<N>) -> Dual<N> {
    let s = x.value.sqrt();
    x.chain(s, 0.5 / s)
}

impl<N: Size> Real for Dual<N> {
    fn __expand_constant(scope: &Scope, x: NativeExpand<f32>) -> DualExpand<N> {
        dual_constant::expand::<N>(scope, x)
    }
    fn __expand_exp(scope: &Scope, x: DualExpand<N>) -> DualExpand<N> {
        dual_exp::expand::<N>(scope, x)
    }
    fn __expand_ln(scope: &Scope, x: DualExpand<N>) -> DualExpand<N> {
        dual_ln::expand::<N>(scope, x)
    }
    fn __expand_sin(scope: &Scope, x: DualExpand<N>) -> DualExpand<N> {
        dual_sin::expand::<N>(scope, x)
    }
    fn __expand_cos(scope: &Scope, x: DualExpand<N>) -> DualExpand<N> {
        dual_cos::expand::<N>(scope, x)
    }
    fn __expand_sqrt(scope: &Scope, x: DualExpand<N>) -> DualExpand<N> {
        dual_sqrt::expand::<N>(scope, x)
    }
}
