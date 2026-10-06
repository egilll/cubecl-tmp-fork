use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
struct Quantity<Tag: 'static, T: CubePrimitive>(T, #[cube(comptime)] PhantomData<Tag>);

struct Distance;

#[cube(launch)]
fn copy(
    input: &Storage<Quantity<Distance, f32>, ReadOnly>,
    output: &mut Storage<Quantity<Distance, f32>>,
) {
    if ABSOLUTE_POS < output.len() {
        output.store(ABSOLUTE_POS, input.load(ABSOLUTE_POS));
    }
}

fn main() {
    let _ = Quantity::<Distance, f32>::from_repr(1.0);
}
