//! What cube barriers cost on the host, measured on demand with `--ignored` in a release build:
//! a stencil that exchanges values through shared memory with two barriers per round, against
//! the same arithmetic reading its neighbours from global memory without barriers.

use std::time::Instant;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;

const ROUNDS: u32 = 8;

#[cube(launch)]
fn exchange_through_shared(input: &[f32], out: &mut [f32], #[comptime] units: usize) {
    let mut tile = Shared::<[f32]>::new_slice(units);
    let mut value = input[ABSOLUTE_POS];
    for _ in 0..ROUNDS {
        tile[UNIT_POS] = value;
        sync_cube();
        let next = tile[(UNIT_POS + 1) % units];
        let previous = tile[(UNIT_POS + units - 1) % units];
        value = 0.5 * value + 0.25 * (next + previous);
        sync_cube();
    }
    out[ABSOLUTE_POS] = value;
}

#[cube(launch)]
fn exchange_through_global(input: &[f32], out: &mut [f32], #[comptime] units: usize) {
    let base = CUBE_POS * units;
    let mut value = input[ABSOLUTE_POS];
    for _ in 0..ROUNDS {
        let next = input[base + (UNIT_POS + 1) % units];
        let previous = input[base + (UNIT_POS + units - 1) % units];
        value = 0.5 * value + 0.25 * (next + previous);
    }
    out[ABSOLUTE_POS] = value;
}

/// Seconds per launch, the median of several after one that compiles.
fn time(client: &Client, launch: impl Fn()) -> f64 {
    launch();
    cubecl::future::block_on(client.sync()).unwrap();
    let mut samples: Vec<f64> = (0..7)
        .map(|_| {
            let start = Instant::now();
            launch();
            cubecl::future::block_on(client.sync()).unwrap();
            start.elapsed().as_secs_f64()
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "a measurement, run on demand"]
fn barrier_throughput() {
    let client = CpuRuntime::client(&Default::default());
    let total = 1 << 22;
    for units in [64u32, 128, 256, 1024] {
        let cubes = total / units;
        let input: Vec<f32> = (0..total).map(|i| (i % 97) as f32).collect();
        let input = client.create_from_slice(f32::as_bytes(&input));
        let out = client.empty(total as usize * core::mem::size_of::<f32>());
        let args = || unsafe {
            (
                BufferArg::from_raw_parts(input.clone(), total as usize),
                BufferArg::from_raw_parts(out.clone(), total as usize),
            )
        };
        let shared = time(&client, || {
            let (input, out) = args();
            exchange_through_shared::launch(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(units),
                input,
                out,
                units as usize,
            );
        });
        let global = time(&client, || {
            let (input, out) = args();
            exchange_through_global::launch(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d(units),
                input,
                out,
                units as usize,
            );
        });
        let per_unit_round = |seconds: f64| seconds / f64::from(total) / f64::from(ROUNDS) * 1e9;
        println!(
            "{units} units per cube: {:.2} ns per unit and round with barriers, {:.2} ns without",
            per_unit_round(shared),
            per_unit_round(global),
        );
    }
}
