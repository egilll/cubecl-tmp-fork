use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, key)]
struct CellId(u32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, key)]
struct TriangleId(u32);

#[cube]
fn area(areas: &Storage<f32, ReadOnly, CellId>, triangle: TriangleId) -> f32 {
    areas.load(triangle)
}

fn main() {}
