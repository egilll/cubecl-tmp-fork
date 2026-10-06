use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, CubeTypeMut)]
struct Distance(f32);

#[derive(CubeType)]
struct Time(f32);

#[derive(CubeType)]
struct Speed(f32);

#[cube]
impl core::ops::Div<Time> for Distance {
    type Output = Speed;
    fn div(self, rhs: Time) -> Speed {
        Speed(self.0 / rhs.0)
    }
}

#[cube]
trait Length: CubeType + core::ops::Div<Time, Output = Speed> {}

impl Length for Distance {}

#[cube]
fn speed<L: Length>(length: L, time: Time) -> Speed {
    length / time
}

#[cube]
fn add<T: CubeType + core::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

#[cube]
fn add_where<T: CubeType>(a: T, b: T) -> T
where
    T: core::ops::Add<Output = T>,
{
    a + b
}

#[cube(launch)]
fn kernel(output: &mut [f32]) {
    output[0] = speed::<Distance>(Distance(6.0), Time(2.0)).0;
    output[1] = add::<f32>(1.0f32, add_where::<f32>(2.0f32, 3.0f32));
}

fn main() {}

#[cube(default_methods)]
trait Additive: CubeType + core::ops::Add<Output = Self> + Sized {
    fn combine(self, rhs: Self) -> Self {
        self + rhs
    }

    fn sum(a: Self, b: Self) -> Self {
        a + b
    }
}

#[cube(default_methods)]
impl Additive for f32 {}

#[cube]
fn defaults<T: Additive>(a: T, b: T) -> T {
    T::sum(a, b)
}

#[cube]
impl core::ops::AddAssign<f32> for Distance {
    fn add_assign(&mut self, rhs: f32) {
        self.0 += rhs;
    }
}

#[cube]
fn offset<T: CubeType + core::ops::AddAssign<f32>>(mut value: T, rhs: f32) -> T {
    value += rhs;
    value
}

#[cube]
fn divide<L: CubeType, R: CubeType>(lhs: L, rhs: R) -> <L as core::ops::Div<R>>::Output
where
    L: core::ops::Div<R>,
    <L as core::ops::Div<R>>::Output: CubeType,
{
    lhs / rhs
}

#[cube]
fn compare<T: CubeType + core::cmp::PartialOrd>(lhs: T, rhs: T) -> bool {
    lhs == rhs || lhs < rhs
}

#[cube]
fn refs(lhs: &f32, rhs: &f32) -> f32 {
    lhs + rhs
}

#[cube]
trait Quotient: CubeType
where
    Self: core::ops::Div<Self::Duration, Output = Self::Velocity>,
{
    type Duration: CubeType;
    type Velocity: CubeType;
    fn velocity(self, duration: Self::Duration) -> Self::Velocity;
}

#[cube]
impl Quotient for Distance {
    type Duration = Time;
    type Velocity = Speed;
    fn velocity(self, duration: Time) -> Speed {
        self / duration
    }
}

#[cube]
fn algebraic_cases(output: &mut [f32]) {
    output[0] = offset::<Distance>(Distance(2.0), 1.0).0;
    output[1] = divide::<Distance, Time>(Distance(6.0), Time(2.0)).0;
    output[2] = Distance(6.0).velocity(Time(2.0)).0;
    output[3] = compare::<f32>(1.0, 2.0) as u32 as f32;
    let lhs = output[0];
    let rhs = output[1];
    output[4] = refs(&lhs, &rhs);
}

#[cube(default_methods)]
trait StaticAdditive: CubeType + core::ops::Add<Output = Self> + Sized {
    fn sum(a: Self, b: Self) -> Self {
        a + b
    }
}

#[cube(default_methods)]
impl StaticAdditive for f32 {}

#[cube]
fn static_default<T: StaticAdditive>(a: T, b: T) -> T {
    T::sum(a, b)
}

trait Dimension: 'static {}
trait Per<D: Dimension>: Dimension {
    type Quotient: Dimension;
}
struct Metre;
struct Second;
struct MetrePerSecond;
impl Dimension for Metre {}
impl Dimension for Second {}
impl Dimension for MetrePerSecond {}
impl Per<Second> for Metre {
    type Quotient = MetrePerSecond;
}

#[derive(CubeType)]
struct Quantity<D: Dimension>(f32, #[cube(comptime)] core::marker::PhantomData<D>);

#[cube]
impl<A: Dimension, B: Dimension> core::ops::Div<Quantity<B>> for Quantity<A>
where
    A: Per<B>,
{
    type Output = Quantity<A::Quotient>;
    fn div(self, rhs: Quantity<B>) -> Quantity<A::Quotient> {
        Quantity::<A::Quotient>(self.0 / rhs.0, comptime! { core::marker::PhantomData })
    }
}

#[cube]
fn velocity(length: Quantity<Metre>, duration: Quantity<Second>) -> Quantity<MetrePerSecond> {
    length / duration
}
