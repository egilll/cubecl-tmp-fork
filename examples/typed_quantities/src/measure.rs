use super::*;
use std::time::{Duration, Instant};

#[cube(inline(never))]
fn raw_velocity(length: f32, duration: f32) -> f32 {
    length / duration
}

#[cube(launch)]
fn raw_calculate(input: &[f32], duration: &[f32], output: &mut [f32]) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output[index] = raw_velocity(input[index], duration[index]);
    }
}

fn time(client: &Client, launch: impl Fn(), iterations: u32) -> Duration {
    cubecl::future::block_on(client.sync()).unwrap();
    let start = Instant::now();
    for _ in 0..iterations {
        launch();
    }
    cubecl::future::block_on(client.sync()).unwrap();
    start.elapsed() / iterations
}

pub fn run(client: &Client) {
    let len = 65_536;
    let input = Buffer::create(client, &vec![6.0f32; len]);
    let times = Buffer::create(client, &vec![2.0f32; len]);
    let raw_output = Buffer::<f32>::empty(client, len);
    let typed_input = StorageBuffer::<Meters>::from_native(input.clone());
    let typed_times = StorageBuffer::<Seconds>::from_native(times.clone());
    let typed_output = StorageBuffer::<Speed>::empty(client, len);
    let count = CubeCount::Static(len as u32 / 256, 1, 1);
    let dim = CubeDim::new_1d(256);
    let raw = || {
        raw_calculate::launch(
            client,
            count.clone(),
            dim,
            (&input).into(),
            (&times).into(),
            (&raw_output).into(),
        );
    };
    let typed = || {
        calculate::launch(
            client,
            count.clone(),
            dim,
            (&typed_input).into(),
            (&typed_times).into(),
            (&typed_output).into(),
        );
    };
    let raw_first = time(client, raw, 1);
    let typed_first = time(client, typed, 1);
    let raw_warm = time(client, raw, 100);
    let typed_warm = time(client, typed, 100);
    let raw_values = raw_output.read(client).unwrap();
    let typed_values: Vec<_> = typed_output
        .read(client)
        .unwrap()
        .into_iter()
        .map(|value| value.0)
        .collect();
    assert_eq!(raw_values, typed_values);
    assert!(raw_values.iter().all(|value| *value == 3.0));
    println!(
        "First launch + sync (includes compilation/cache): raw {raw_first:?}, typed {typed_first:?}"
    );
    println!("100 launches + sync, mean wall time: raw {raw_warm:?}, typed {typed_warm:?}");
}
