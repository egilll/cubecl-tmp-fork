//! Accurate arithmetic without `f64`, for targets that lack it (Metal, most
//! of WebGPU).
//!
//! - [`Compensated`] is a Neumaier sum: an `f32` total and a running
//!   correction, for long accumulations whose result must not drift.
//! - [`DoubleF32`] is an unevaluated sum `hi + lo` of two `f32`s, which
//!   carries about 48 bits of significand through addition and
//!   multiplication.
//!
//! Both rely on every operation being rounded on its own: no reassociation,
//! no contraction. That is CubeCL's default (precise MSL math, `NoContraction`
//! on SPIR-V); don't use them inside `fast_math`. WGSL makes no promise
//! either way.

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// `(s, e)` with `s = fl(a + b)` and `a + b = s + e` exactly (Knuth's TwoSum).
#[cube]
pub fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `(p, e)` with `p = fl(a * b)` and `a * b = p + e` exactly, through `fma`.
#[cube]
pub fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let e = fma(a, b, -p);
    (p, e)
}

/// A Neumaier-compensated `f32` sum.
#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug)]
#[expand(derive(Clone, Copy))]
pub struct Compensated {
    sum: f32,
    correction: f32,
}

#[cube]
impl Compensated {
    /// An empty sum.
    pub fn new() -> Compensated {
        Compensated {
            sum: 0.0,
            correction: 0.0,
        }
    }

    /// Add `x`, keeping what rounding lost in the correction.
    pub fn add(&mut self, x: f32) {
        let t = self.sum + x;
        let lost = select(
            f32::abs(self.sum) >= f32::abs(x),
            (self.sum - t) + x,
            (x - t) + self.sum,
        );
        self.correction += lost;
        self.sum = t;
    }

    /// The compensated total.
    pub fn value(&self) -> f32 {
        self.sum + self.correction
    }
}

/// An unevaluated sum `hi + lo` of two `f32`s, `|lo| <= ulp(hi) / 2`.
#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug, PartialEq)]
#[expand(derive(Clone, Copy))]
pub struct DoubleF32 {
    pub hi: f32,
    pub lo: f32,
}

impl DoubleF32 {
    /// The nearest double-f32 to a host `f64`.
    pub fn from_f64(x: f64) -> DoubleF32 {
        let hi = x as f32;
        DoubleF32 {
            hi,
            lo: (x - hi as f64) as f32,
        }
    }

    /// Its value as a host `f64`.
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
}

#[cube]
impl DoubleF32 {
    /// `x` exactly.
    pub fn from_f32(x: f32) -> DoubleF32 {
        DoubleF32 { hi: x, lo: 0.0 }
    }

    /// Renormalize `hi + lo` so `lo` is below half an ulp of `hi`.
    fn normalized(hi: f32, lo: f32) -> DoubleF32 {
        let (s, e) = two_sum(hi, lo);
        DoubleF32 { hi: s, lo: e }
    }

    /// The nearest `f32`.
    pub fn to_f32(&self) -> f32 {
        self.hi + self.lo
    }
}

#[cube]
impl core::ops::Add for DoubleF32 {
    type Output = DoubleF32;
    fn add(self, rhs: DoubleF32) -> DoubleF32 {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let (t, f) = two_sum(self.lo, rhs.lo);
        let e = e + t;
        let (s, e) = two_sum(s, e);
        let e = e + f;
        DoubleF32::normalized(s, e)
    }
}

#[cube]
impl core::ops::Neg for DoubleF32 {
    type Output = DoubleF32;
    fn neg(self) -> DoubleF32 {
        DoubleF32 {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

#[cube]
impl core::ops::Sub for DoubleF32 {
    type Output = DoubleF32;
    fn sub(self, rhs: DoubleF32) -> DoubleF32 {
        self + -rhs
    }
}

#[cube]
impl core::ops::Mul for DoubleF32 {
    type Output = DoubleF32;
    fn mul(self, rhs: DoubleF32) -> DoubleF32 {
        let (p, e) = two_prod(self.hi, rhs.hi);
        let e = e + (self.hi * rhs.lo + self.lo * rhs.hi);
        DoubleF32::normalized(p, e)
    }
}
