use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
struct Invalid(f32, ());

fn main() {}
