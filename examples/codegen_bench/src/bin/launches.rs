//! Host-side launch overhead: many tiny launches, the shape of an application
//! that runs short event rounds and reads results back between them.
use cubecl::prelude::*;
use std::time::{Duration, Instant};

#[cube(launch)]
fn axpy(x: &[f32], y: &mut [f32], a: f32) {
    let i = ABSOLUTE_POS;
    if i < y.len() {
        y[i] += a * x[i];
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let client = cubecl::Device::default().client();
    let n: usize = std::env::args()
        .nth(1)
        .map(|d| d.parse().unwrap())
        .unwrap_or(1024);
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

    // Enqueue only: launches back to back, one sync at the end.
    let mut enqueue = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for i in 0..5000 {
            launch(i as f32 * 1e-6);
        }
        let enq = start.elapsed();
        cubecl::future::block_on(client.sync()).unwrap();
        enqueue.push((enq, start.elapsed()));
    }
    let enq = median(enqueue.iter().map(|e| e.0).collect());
    let total = median(enqueue.iter().map(|e| e.1).collect());

    // Round trips: launch then sync, as a loop that waits on each round.
    let mut sync = Vec::new();
    for _ in 0..500 {
        let start = Instant::now();
        launch(1e-6);
        cubecl::future::block_on(client.sync()).unwrap();
        sync.push(start.elapsed());
    }

    // Round trips with a small read back of the result.
    let mut read = Vec::new();
    for _ in 0..500 {
        let start = Instant::now();
        launch(1e-6);
        let _ = client.read_one(y.clone()).unwrap();
        read.push(start.elapsed());
    }

    println!(
        "{} n={n} enqueue/launch {:?} (5000 launches + sync {:?}) launch+sync {:?} launch+read {:?}",
        client.name(),
        enq / 5000,
        total,
        median(sync),
        median(read)
    );
}
