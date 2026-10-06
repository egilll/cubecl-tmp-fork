use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};
use cubecl_runtime::runtime::Runtime;

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr, Clone, Copy, Debug, PartialEq)]
struct Distance(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone, Copy)]
struct Time(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone, Copy, Debug, PartialEq)]
struct Speed(f32);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
struct Point<F: 'static>(Vector<f32, Const<3>>, #[cube(comptime)] PhantomData<F>);

struct Room;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, key)]
struct Cell(u32);

#[cube]
impl core::ops::Div<Time> for Distance {
    type Output = Speed;
    fn div(self, rhs: Time) -> Speed {
        Speed(self.0 / rhs.0)
    }
}

#[cube]
trait Length: CubeType + core::ops::Div<Time, Output = Speed> {}

impl Length for Distance {}

#[cube(inline(never))]
fn speed<L: Length>(distance: L, time: Time) -> Speed {
    distance / time
}

#[cube(inline(never))]
fn translate<F: 'static>(point: Point<F>, distance: f32) -> Point<F> {
    Point::<F>(point.0 + Vector::new(distance), comptime! { PhantomData })
}

#[cube(launch)]
fn kernel_speed(
    input: &Storage<Distance, ReadOnly>,
    times: &Storage<Time, ReadOnly>,
    output: &mut Storage<Speed>,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let point = Point::<Room>(Vector::new(input.load(index).0), comptime! { PhantomData });
        let mut distance = Distance(translate(point, 1.0).0.extract(0usize));
        distance.0 = distance.0 - 1.0;
        output.store(index, speed::<Distance>(distance, times.load(index)));
    }
}

#[cube(launch)]
fn kernel_copy(input: &Storage<Distance, ReadOnly>, output: &mut Storage<Distance>) {
    if ABSOLUTE_POS < output.len() {
        output.store(ABSOLUTE_POS, input.load(ABSOLUTE_POS));
    }
}

#[cube(launch)]
fn kernel_gather(
    table: &Storage<Distance, ReadOnly, Cell>,
    cells: &Storage<Cell, ReadOnly>,
    output: &mut Storage<Distance>,
) {
    if ABSOLUTE_POS == 0 {
        output.fill(Distance(-1.0));
        output.gather(table, cells);
    }
}

#[cube(launch)]
fn kernel_load_snapshot(output: &mut Storage<Distance>) {
    if ABSOLUTE_POS == 0 {
        let old = output.load(0);
        output.store(0, Distance(99.0));
        output.store(1, old);
    }
}

pub fn test_storage_round_trip<R: Runtime>(client: Client) {
    let distances: Vec<_> = (0..64).map(|i| Distance(i as f32 - 32.0)).collect();
    let times = [Time(2.0); 64];
    let input = StorageBuffer::create(&client, &distances);
    let times = StorageBuffer::create(&client, &times);
    let output = StorageBuffer::<Speed>::empty(&client, 64);
    kernel_speed::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        (&input).into(),
        (&times).into(),
        (&output).into(),
    );
    let values = output.read(&client).unwrap();
    for (index, value) in values.iter().enumerate() {
        assert_eq!(*value, Speed(distances[index].0 / 2.0));
    }
    let window = input.slice(5..9);
    window.write(&client, &[Distance(123.0), Distance(124.0)]);
    assert_eq!(
        window.read(&client).unwrap(),
        [Distance(123.0), Distance(124.0), distances[7], distances[8]]
    );
    let restored = StorageBuffer::<Distance>::from_native(window.as_native().clone());
    assert_eq!(
        restored.read(&client).unwrap(),
        window.read(&client).unwrap()
    );

    kernel_copy::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        (&input).into(),
        TypedBufferArg::alias(0, 64),
    );
    assert_eq!(input.read(&client).unwrap()[5], Distance(123.0));

    let cells = StorageBuffer::create(&client, &[Cell(8), Cell(6), Cell(0), Cell(5)]);
    let gathered = StorageBuffer::<Distance>::empty(&client, 4);
    kernel_gather::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        (&input).into(),
        (&cells).into(),
        (&gathered).into(),
    );
    assert_eq!(
        gathered.read(&client).unwrap(),
        [distances[8], Distance(124.0), distances[0], Distance(123.0)]
    );
    let snapshot = StorageBuffer::create(&client, &[Distance(3.0), Distance(0.0)]);
    kernel_load_snapshot::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        (&snapshot).into(),
    );
    assert_eq!(
        snapshot.read(&client).unwrap(),
        [Distance(99.0), Distance(3.0)]
    );
}

pub fn test_storage_alias_type<R: Runtime>(client: Client) {
    let input = StorageBuffer::create(&client, &[Distance(4.0)]);
    let times = StorageBuffer::create(&client, &[Time(2.0)]);
    let mut launcher = KernelLauncher::new(KernelSettings::new(
        CubeDim::new_1d(1).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    <Storage<Distance>>::register((&input).into(), &mut launcher);
    <Storage<Time>>::register((&times).into(), &mut launcher);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        <Storage<Speed>>::register(TypedBufferArg::alias(0, 1), &mut launcher);
    }));
    launcher.discard();
    assert!(refused.is_err());
}

#[macro_export]
macro_rules! testgen_storage {
    () => {
        use super::*;
        #[$crate::runtime_tests::test_log::test]
        fn test_storage_round_trip() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_round_trip::<TestRuntime>(client);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_storage_alias_type() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_alias_type::<TestRuntime>(client);
        }
    };
}
