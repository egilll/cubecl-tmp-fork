use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
struct Meters(f32);
#[derive(CubeType)]
struct Seconds(f32);

#[cube]
fn invalid(distance: Meters, duration: Seconds) -> Meters {
    distance + duration
}

fn main() {}
