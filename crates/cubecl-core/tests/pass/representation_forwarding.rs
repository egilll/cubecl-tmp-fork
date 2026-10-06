use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

struct Room;

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
#[device_repr(copy, ord)]
struct Distance<Frame: 'static>(f32, #[cube(comptime)] PhantomData<Frame>);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, eq)]
struct MaterialId(u32);

#[cube]
fn nearest<F: 'static>(a: Distance<F>, b: Distance<F>, limit: Distance<F>) -> Distance<F> {
    let closest = a.min(b);
    if closest < limit && closest != a { closest } else { a.clamp(b, limit) }
}

#[cube]
fn farthest<T: Ordered>(a: T, b: T) -> T {
    a.max(b)
}

#[cube]
fn same(a: MaterialId, b: MaterialId) -> bool {
    a == b
}

#[cube(launch)]
fn kernel(input: &Storage<Distance<Room>, ReadOnly>, output: &mut Storage<Distance<Room>>) {
    let a = input.load(0);
    let b = input.load(1);
    output.store(0, farthest::<Distance<Room>>(nearest::<Room>(a, b, a), b));
}

fn main() {
    let near = Distance::<Room>(1.0, PhantomData);
    let far = Distance::<Room>(2.0, PhantomData);
    assert!(near < far && near == near);
    assert!(MaterialId(3) != MaterialId(4));
}
