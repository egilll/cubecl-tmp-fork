use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, CubeTypeMut, CubeLaunch, IntoRuntime, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Distance(f32);

#[derive(CubeType, CubeTypeMut, CubeLaunch)]
struct Empty();

#[derive(CubeType, CubeTypeMut, CubeLaunch)]
struct Pair<T: CubeType>(T, T);

#[cube]
impl core::ops::Add for Distance {
    type Output = Distance;

    fn add(self, rhs: Distance) -> Distance {
        Distance(self.0 + rhs.0)
    }
}

#[cube(inline(never))]
fn advance(value: &Distance) -> Distance {
    Distance(value.0 + 1.0)
}

#[cube(launch)]
fn kernel(value: Distance, pair: Pair<f32>, output: &mut [f32]) {
    let mut distance = advance(&value);
    distance.0 = distance.0 + pair.0;
    let other = Distance(2.0);
    distance = distance + other;
    let pair = Pair::<f32>(distance.0, pair.1);
    let swapped = Pair::<f32>(pair.1, pair.0);
    output[0] = pair.0 + pair.1 + swapped.0;
}

fn main() {
    let _ = EmptyLaunch::new();
    let _ = DistanceLaunch::new(2.0);
    let _ = PairLaunch::<f32>::new(1.0, 2.0);
}
