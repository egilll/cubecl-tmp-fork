mod measure;

use cubecl::prelude::*;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone)]
struct Meters(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone)]
struct Seconds(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr, Clone)]
struct Speed(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy, key)]
struct CellId(usize);

#[cube]
impl core::ops::Div<Seconds> for Meters {
    type Output = Speed;
    fn div(self, rhs: Seconds) -> Speed {
        Speed(self.0 / rhs.0)
    }
}

#[cube]
trait Length: CubeType + core::ops::Div<Seconds, Output = Speed> {}
impl Length for Meters {}

#[cube(inline(never))]
fn velocity<L: Length>(length: L, duration: Seconds) -> Speed {
    length / duration
}

#[cube(launch)]
fn calculate(
    input: &Storage<Meters, ReadOnly, CellId>,
    duration: &Storage<Seconds, ReadOnly, CellId>,
    output: &mut Storage<Speed>,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let cell = CellId(index);
        output.store(
            index,
            velocity::<Meters>(input.load(cell), duration.load(cell)),
        );
    }
}

fn main() {
    let client = cubecl::Device::default().client();
    let input = StorageBuffer::create(&client, &[Meters(6.0), Meters(12.0)]);
    let times = StorageBuffer::create(&client, &[Seconds(2.0), Seconds(3.0)]);
    let output = StorageBuffer::<Speed>::empty(&client, 2);
    calculate::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        (&input).into(),
        (&times).into(),
        (&output).into(),
    );
    let speeds: Vec<_> = output
        .read(&client)
        .unwrap()
        .into_iter()
        .map(|speed| speed.0)
        .collect();
    assert_eq!(speeds, [3.0, 4.0]);
    println!("Meters / Seconds = {speeds:?} m/s");
    if std::env::args().any(|argument| argument == "--measure") {
        measure::run(&client);
    }
}
