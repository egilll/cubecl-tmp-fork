//! Host enqueue cost on the native Metal runtime, for a profiler to attach to: `enqueue SECS`
//! launches a tiny kernel back to back, syncing every 1000 launches, and prints the mean cost of
//! a launch.
use cubecl::prelude::*;
use std::time::{Duration, Instant};

#[cube(launch)]
fn axpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = ABSOLUTE_POS;
    if i < y.len() {
        y[i] += a * x[i];
    }
}

fn main() {
    let secs: u64 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(3);
    let client = cubecl::Device::metal(Default::default()).unwrap().client();
    let n = 1024;
    let x = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n]));
    let y = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let launch = |a: f32| {
        axpy::launch(
            &client,
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            unsafe { BufferArg::from_raw_parts(x.clone(), n) },
            unsafe { BufferArg::from_raw_parts(y.clone(), n) },
            a,
        )
    };
    launch(0.0);
    cubecl::future::block_on(client.sync()).unwrap();
    let start = Instant::now();
    let (mut launches, mut enqueue) = (0u64, Duration::ZERO);
    while start.elapsed() < Duration::from_secs(secs) {
        let t = Instant::now();
        for i in 0..1000 {
            launch(i as f32 * 1e-6);
        }
        enqueue += t.elapsed();
        launches += 1000;
        cubecl::future::block_on(client.sync()).unwrap();
    }
    println!(
        "{launches} launches, enqueue {:?}/launch",
        enqueue / launches as u32
    );
}
