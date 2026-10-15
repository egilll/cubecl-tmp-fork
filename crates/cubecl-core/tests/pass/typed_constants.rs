use cubecl::prelude::*;
use cubecl_core as cubecl;

/// A typed wrapper around a scalar.
#[derive(CubeType, IntoRuntime, Clone, Copy)]
struct Metres(f32);

const SURFACE: Metres = Metres(1.0e-5);

#[cube]
fn doubled(length: Metres) -> f32 {
    length.0 * 2.0
}

#[cube(launch)]
fn kernel(output: &mut [f32]) {
    let margin = SURFACE;
    output[0] = doubled(margin) + doubled(SURFACE);
}

/// A constant whose methods take runtime arguments.
#[derive(CubeType, IntoRuntime, Clone, Copy)]
struct Interval {
    low: f32,
    high: f32,
}

const UNIT: Interval = Interval { low: 0.0, high: 1.0 };

#[cube]
impl Interval {
    fn locate(self, value: f32) -> f32 {
        (value - self.low) / (self.high - self.low)
    }
}

#[cube(launch)]
fn located(output: &mut [f32]) {
    output[0] = UNIT.locate(output[1]);
}

fn main() {}
