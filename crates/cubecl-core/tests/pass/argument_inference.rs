//! Arguments determine generics in kernels as in Rust: a tuple constructor or
//! a generic function infers its parameters from a kernel value's host type
//! or a literal's own type, even where the operators that use the result
//! would admit another representation.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, CubeTypeMut, CubeLaunch, IntoRuntime, Clone, Copy)]
struct Wide<R: CubeType>(R);

#[derive(CubeType, Clone, Copy)]
struct Pair<R: CubeType>(R, R);

#[derive(CubeType, CubeTypeMut, CubeLaunch, IntoRuntime, Clone, Copy)]
struct Amount<R: CubeType>(R);

#[cube]
impl<R: Float> core::ops::Div for Amount<R> {
    type Output = Amount<R>;
    fn div(self, rhs: Amount<R>) -> Amount<R> {
        Amount(self.0 / rhs.0)
    }
}

/// A second division, by an amount of the pair's scalar, that an argument
/// typed only by its expand type could not tell apart from the first.
#[cube]
impl<R: Float> core::ops::Div<Amount<R>> for Amount<Pair<R>> {
    type Output = Amount<Pair<R>>;
    fn div(self, rhs: Amount<R>) -> Amount<Pair<R>> {
        Amount(Pair(self.0.0 / rhs.0, self.0.1 / rhs.0))
    }
}

#[cube]
fn twice<F: Float>(x: F) -> F {
    x + x
}

#[cube]
fn wrap<T: CubeType>(value: T) -> Wide<T> {
    Wide(value)
}

#[cube(launch)]
fn kernel(amount: Amount<f32>, x: f32, output: &mut [f32]) {
    let a = amount.0;
    let untyped = Amount(1.0) / Amount(a);
    let typed = Amount(2.0f32) / Amount(a);
    let runtime = Amount(x) / Amount(a);
    let bound = Amount(4.0f32);
    output[0] = untyped.0 + typed.0 + runtime.0 + (bound / Amount(a)).0 + Wide::<f32>(a).0;
    output[1] = twice(x) + twice(2.0f32) + wrap(x).0;
}

fn main() {
    let _ = AmountLaunch::<f32>::new(2.0);
    let _ = WideLaunch::<f32>::new(2.0);
}
