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

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy)]
struct Pressure<R: CubePrimitive>(R);

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

#[cube(launch)]
fn kernel_double(values: &mut Storage<Pressure<Vector<f32, Const<4>>>>) {
    if ABSOLUTE_POS < values.len() {
        let value = values.load(ABSOLUTE_POS);
        values.store(ABSOLUTE_POS, Pressure(value.0 + value.0));
    }
}

pub fn test_storage_vectors<R: Runtime>(client: Client) {
    let values: Vec<_> = (0..8).map(|i| Pressure(i as f32)).collect();
    let pressures = StorageBuffer::create(&client, &values);
    kernel_double::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        pressures.vectors::<4>(),
    );
    let doubled: Vec<_> = pressures.read(&client).unwrap().iter().map(|p| p.0).collect();
    assert_eq!(doubled, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    let misaligned = std::panic::catch_unwind(|| pressures.slice(1..5).vectors::<4>());
    assert!(misaligned.is_err());
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

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr)]
#[device_repr(copy, packed)]
struct Packed3<F: 'static>(Vector<f32, Const<3>>, #[cube(comptime)] PhantomData<F>);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, key)]
struct Source(u32);

/// Cell-major interleaving of per-source values.
#[derive(CubeType)]
struct CellSource {
    cell: Cell,
    source: Source,
    sources: u32,
}

#[cube]
impl StorageKey for CellSource {
    fn position(key: CellSource) -> usize {
        Cell::position(key.cell) * key.sources as usize + Source::position(key.source)
    }
}

/// Typed tables as members of a launched struct.
#[derive(CubeType, CubeLaunch)]
struct Field {
    points: Storage<Packed3<Room>, ReadOnly, Cell>,
    values: Storage<Distance, ReadWrite, CellSource>,
    sources: u32,
}

#[cube(launch)]
fn kernel_packed(field: &mut Field, output: &mut Storage<Packed3<Room>, ReadWrite, Cell>) {
    let cell = ABSOLUTE_POS as u32;
    if (cell as usize) < output.len() {
        let point = field.points.load(Cell(cell));
        let lift = Vector::new(10.0);
        output.store(Cell(cell), Packed3::<Room>(point.0 + lift, comptime! { PhantomData }));
        let mut scratch = LocalStorage::<Packed3<Room>, Source>::new(2usize);
        scratch.store(Source(1), point);
        let mut source = 0u32;
        while source < field.sources {
            let key = CellSource {
                cell: Cell(cell),
                source: Source(source),
                sources: field.sources,
            };
            field
                .values
                .store(key, Distance(scratch.load(Source(1)).0.extract(0usize) + source as f32));
            source += 1;
        }
    }
}

pub fn test_storage_packed<R: Runtime>(client: Client) {
    let points: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let points = StorageBuffer::<Packed3<Room>>::from_elements(&client, &points);
    assert_eq!(points.len(), 4);
    let output = StorageBuffer::<Packed3<Room>>::empty(&client, 4);
    let values = StorageBuffer::<Distance>::empty(&client, 8);
    kernel_packed::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        FieldLaunch::new((&points).into(), (&values).into(), 2),
        (&output).into(),
    );
    let lifted = output.read_elements(&client).unwrap();
    let expected: Vec<f32> = (0..12).map(|i| i as f32 + 10.0).collect();
    assert_eq!(lifted, expected);
    assert_eq!(
        output.slice(1..3).read_elements(&client).unwrap(),
        expected[3..9]
    );
    let values = values.read(&client).unwrap();
    let expected: Vec<_> = (0..4)
        .flat_map(|cell| [0.0, 1.0].map(|source| Distance(cell as f32 * 3.0 + source)))
        .collect();
    assert_eq!(values, expected);
}

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Debug)]
#[device_repr(copy, ord)]
struct Count(u32);

#[cube]
impl core::ops::Add for Count {
    type Output = Count;
    fn add(self, rhs: Count) -> Count {
        Count(self.0 + rhs.0)
    }
}

#[cube(launch)]
fn kernel_atomic(counts: &AtomicStorage<Count, Cell>, peaks: &AtomicStorage<Count>) {
    let cell = Cell(ABSOLUTE_POS as u32 % 2);
    counts.fetch_add(cell, Count(1));
    peaks.fetch_max(0usize, Count(ABSOLUTE_POS as u32));
    peaks.fetch_min(1usize, Count(ABSOLUTE_POS as u32));
}

pub fn test_storage_atomic<R: Runtime>(client: Client) {
    let counts = StorageBuffer::create(&client, &[Count(0), Count(5)]);
    let peaks = StorageBuffer::create(&client, &[Count(0), Count(u32::MAX)]);
    kernel_atomic::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        (&counts).into(),
        (&peaks).into(),
    );
    assert_eq!(counts.read(&client).unwrap(), [Count(16), Count(21)]);
    assert_eq!(peaks.read(&client).unwrap(), [Count(31), Count(0)]);
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
        fn test_storage_vectors() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_vectors::<TestRuntime>(client);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_storage_alias_type() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_alias_type::<TestRuntime>(client);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_storage_packed() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_packed::<TestRuntime>(client);
        }
        #[$crate::runtime_tests::test_log::test]
        fn test_storage_atomic() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::storage::test_storage_atomic::<TestRuntime>(client);
        }
    };
}
