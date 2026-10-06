use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

struct Room;
struct World;
#[derive(CubeType)]
struct Position<F: 'static>(f32, #[cube(comptime)] PhantomData<F>);

fn invalid(world: Position<World>) -> Position<Room> {
    world
}

fn main() {}
