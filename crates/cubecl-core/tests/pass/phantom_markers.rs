use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

struct Room;
struct World;

#[derive(CubeType, CubeTypeMut, CubeLaunch, IntoRuntime)]
struct Position<Frame: 'static>(f32, #[cube(comptime)] PhantomData<Frame>);

#[cube(inline(never))]
fn translate<F: 'static>(point: Position<F>) -> Position<F> {
    Position::<F>(point.0 + 1.0, comptime! { PhantomData })
}

#[cube(launch)]
fn kernel(room: Position<Room>, world: Position<World>, output: &mut [f32]) {
    let mut point = translate(room);
    point = translate(point);
    output[0] = point.0 + translate(world).0;
}

fn main() {
    let _ = PositionLaunch::<Room>::new(1.0);
    let _ = PositionLaunch::<World>::new(2.0);
}
