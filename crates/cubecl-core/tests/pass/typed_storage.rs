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

#[repr(transparent)]
#[derive(CubeType, CubeLaunch, DeviceRepr)]
struct Branded<Tag: Send + Sync + 'static>(f32, #[cube(comptime)] PhantomData<Tag>);

#[cube(launch)]
fn scale(factor: Branded<Distance>, output: &mut Storage<Quantity<Distance, f32>>) {
    output.store(0, Quantity::<Distance, f32>(factor.0, comptime! { PhantomData }));
}

fn launch_takes_no_marker() {
    let _ = BrandedLaunch::<Distance>::new(2.0);
}
