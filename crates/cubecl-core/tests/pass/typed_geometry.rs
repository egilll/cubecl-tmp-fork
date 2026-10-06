use core::marker::PhantomData;
use cubecl::prelude::*;
use cubecl_core as cubecl;

struct Room;
struct World;

type Vec3 = Vector<f32, Const<3>>;

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
#[device_repr(copy)]
struct Point<F: 'static>(Vec3, #[cube(comptime)] PhantomData<F>);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
#[device_repr(copy)]
struct Displacement<F: 'static>(Vec3, #[cube(comptime)] PhantomData<F>);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, CubeLaunch, DeviceRepr)]
#[device_repr(copy, ord)]
struct Meters(f32);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
#[device_repr(copy, eq, key)]
struct TriangleId(u32);

#[derive(CubeType, CubeTypeMut)]
struct Hit {
    found: bool,
    distance: Meters,
    triangle: TriangleId,
}

#[cube]
impl<F: 'static> core::ops::Sub for Point<F> {
    type Output = Displacement<F>;
    fn sub(self, rhs: Point<F>) -> Displacement<F> {
        Displacement(self.0 - rhs.0, comptime! { PhantomData })
    }
}

#[cube]
impl<F: 'static> core::ops::Add<Displacement<F>> for Point<F> {
    type Output = Point<F>;
    fn add(self, rhs: Displacement<F>) -> Point<F> {
        Point(self.0 + rhs.0, comptime! { PhantomData })
    }
}

#[derive(CubeType, CubeLaunch)]
struct RoomToWorld {
    height: f32,
}

#[cube]
impl RoomToWorld {
    fn apply(&self, point: Point<Room>) -> Point<World> {
        let lift = Vec3::new(self.height);
        Point::<World>(point.0 + lift, comptime! { PhantomData })
    }
}

#[cube]
fn closest(hit: Hit, distance: Meters, triangle: TriangleId) -> Hit {
    let mut hit = hit;
    if !hit.found || distance < hit.distance {
        hit = Hit {
            found: true,
            distance,
            triangle,
        };
    }
    hit
}

#[cube(launch)]
fn trace(
    max_distance: Meters,
    transform: RoomToWorld,
    areas: &Storage<f32, ReadOnly, TriangleId>,
    sources: &Storage<Point<Room>, ReadOnly>,
    output: &mut Storage<Point<World>>,
) {
    let source = sources.load(ABSOLUTE_POS);
    let step = source - sources.load(0);
    let hit = Hit {
        found: false,
        distance: max_distance,
        triangle: TriangleId(0),
    };
    let hit = closest(hit, Meters(areas.load(TriangleId(1))), TriangleId(1));
    if hit.found && hit.triangle != TriangleId(0) {
        output.store(ABSOLUTE_POS, transform.apply(source + step));
    }
}

fn main() {
    let _ = MetersLaunch::new(1.0);
    let _ = RoomToWorldLaunch::new(2.5);
}
