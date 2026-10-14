//! What argument inference reaches beyond direct arguments: a literal bound
//! to a name, a generic function's primitive result, a reference to a
//! generic, a method's own generics, and named struct literals, all as in
//! Rust, beside a second `Div` impl that a projection could not tell apart.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Amount<R: CubeType> {
    value: R,
}

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Pair<R: CubeType> {
    first: R,
    second: R,
}

#[cube]
impl<R: Float> core::ops::Div for Amount<R> {
    type Output = Amount<R>;
    fn div(self, rhs: Amount<R>) -> Amount<R> {
        Amount::<R> {
            value: self.value / rhs.value,
        }
    }
}

#[cube]
impl<R: Float> core::ops::Div<Amount<R>> for Amount<Pair<R>> {
    type Output = Amount<Pair<R>>;
    fn div(self, rhs: Amount<R>) -> Amount<Pair<R>> {
        let pair = Pair::<R> {
            first: self.value.first / rhs.value,
            second: self.value.second / rhs.value,
        };
        Amount::<Pair<R>> { value: pair }
    }
}

#[cube]
fn twice<F: Float>(x: F) -> F {
    x + x
}

#[cube]
fn copied<T: CubePrimitive>(value: &T) -> T {
    *value
}

#[derive(CubeType, Clone, Copy)]
struct Scale {
    factor: f32,
}

#[cube]
impl Scale {
    fn apply<F: Float>(&self, x: F, output: &mut Array<F>) {
        output[0] = x * F::cast_from(self.factor);
    }
}

/// A generic expand type copies where its fields' expand types do.
fn copies<T: Copy>(value: T) -> (T, T) {
    (value, value)
}

#[cube]
fn kernel(x: f32, scale: Scale, output: &mut Array<f32>) {
    let k = 2.0;
    let mut sum = 0.0;
    sum += x;
    let named = Amount { value: 1.0 } / Amount { value: x };
    let pair = Pair { second: x, first: k };
    output[1] = k * x + twice(2.0) + x + copied(&x) + sum + named.value + pair.first;
    scale.apply(x, output);
}

fn main() {
    let _ = kernel::expand;
    let _ = copies::<AmountExpand<f32>>;
}
