use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, ord)]
struct Distance(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, ord)]
struct Time(f32);

#[cube]
fn compare(distance: Distance, time: Time) -> bool {
    distance < time
}

fn main() {}
