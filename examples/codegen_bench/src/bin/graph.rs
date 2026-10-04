//! Launch graphs on the native Metal runtime: `graph N K` runs rounds of K small launches and one
//! sync (`graph N K ITERS`), either independent (each writes its own buffer) or a chain (each updates the same one),
//! and prints the median round. Shows what dispatches overlapping on the GPU is worth.
use cubecl::prelude::*;
use std::time::{Duration, Instant};

#[cube(launch)]
fn smooth(x: &[f32], y: &mut [f32], a: f32, iters: u32) {
    let i = ABSOLUTE_POS;
    if i < y.len() {
        let mut acc = y[i];
        for k in 0..iters as usize {
            acc = acc * 0.99 + a * x[(i + k) % x.len()];
        }
        y[i] = acc;
    }
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn main() {
    let client = cubecl::Device::metal(Default::default()).unwrap().client();
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>().unwrap());
    let n = args.next().unwrap_or(4096);
    let k = args.next().unwrap_or(64);
    let iters = args.next().unwrap_or(64) as u32;
    let x = client.create_from_slice(f32::as_bytes(&vec![1.0f32; n]));
    let ys: Vec<_> = (0..k)
        .map(|_| client.create_from_slice(f32::as_bytes(&vec![0.0f32; n])))
        .collect();
    let launch = |y: &cubecl::server::Handle, a: f32| {
        smooth::launch(
            &client,
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            unsafe { BufferArg::from_raw_parts(x.clone(), n) },
            unsafe { BufferArg::from_raw_parts(y.clone(), n) },
            a,
            iters,
        )
    };
    // Clocks: the GPU ramps up over about a second of steady work, and the first
    // process to run after a pause would otherwise measure its power state.
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_secs(2) {
        for y in &ys {
            launch(y, 0.0);
        }
        cubecl::future::block_on(client.sync()).unwrap();
    }
    for (name, chain, serial) in [
        ("independent", false, false),
        ("independent serial", false, true),
        ("chain", true, false),
        ("chain serial", true, true),
        ("independent", false, false),
        ("independent serial", false, true),
    ] {
        // Read by a measurement-only build of cubecl-metal, at each new encoder.
        unsafe {
            if serial {
                std::env::set_var("AB_SERIAL", "1")
            } else {
                std::env::remove_var("AB_SERIAL")
            }
        }
        let mut rounds = Vec::new();
        for r in 0..40 {
            let start = Instant::now();
            for (i, y) in ys.iter().enumerate() {
                launch(if chain { &ys[0] } else { y }, (r * k + i) as f32 * 1e-6);
            }
            cubecl::future::block_on(client.sync()).unwrap();
            rounds.push(start.elapsed());
        }
        let check: f32 = f32::from_bytes(&client.read_one(ys[0].clone()).unwrap())
            .iter()
            .sum();
        println!(
            "n={n} k={k} iters={iters} {name:>18}: round {:?} ({:?}/launch), checksum {check:.6e}",
            median(rounds.clone()),
            median(rounds) / k as u32
        );
    }
}
