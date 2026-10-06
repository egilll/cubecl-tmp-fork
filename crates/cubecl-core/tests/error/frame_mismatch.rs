use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

struct Room;
struct World;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy)]
struct Point<F: 'static>(f32, #[cube(comptime)] PhantomData<F>);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy)]
struct Displacement<F: 'static>(f32, #[cube(comptime)] PhantomData<F>);

#[cube]
impl<F: 'static> core::ops::Add<Displacement<F>> for Point<F> {
    type Output = Point<F>;
    fn add(self, rhs: Displacement<F>) -> Point<F> {
        Point(self.0 + rhs.0, comptime! { PhantomData })
    }
}

#[cube]
fn moved(point: Point<Room>, step: Displacement<World>) -> Point<Room> {
    point + step
}

#[cube]
fn summed(a: Point<Room>, b: Point<Room>) -> Point<Room> {
    a + b
}

fn main() {}
