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

fn main() {}
