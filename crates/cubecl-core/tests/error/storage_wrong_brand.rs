use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone)]
struct Distance(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone)]
struct Time(f32);

fn bind(buffer: &StorageBuffer<Time>) -> TypedBufferArg<Distance> {
    buffer.into()
}

fn main() {}
