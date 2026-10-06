use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(packed)]
struct Metres(f32);

#[cube]
fn load(lengths: &Storage<Metres, ReadOnly>) -> Metres {
    lengths.load(0)
}

fn main() {}
