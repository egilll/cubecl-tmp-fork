//! Ordering under the concurrent encoder: dispatches in one batch may overlap unless the launch
//! inserts a barrier, so each test chains work through memory and checks the exact result
//! against the CPU. A missed hazard shows up as a lost update.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn step_kernel(values: &mut [u32]) {
    if ABSOLUTE_POS < values.len() {
        values[ABSOLUTE_POS] = values[ABSOLUTE_POS] * 3 + 1;
    }
}

#[cube(launch_unchecked)]
fn step_into_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * 3 + 1;
    }
}

#[cube(launch_unchecked)]
fn accumulate_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] += input[ABSOLUTE_POS];
    }
}

const N: usize = 4096;

fn step(x: u32) -> u32 {
    x.wrapping_mul(3).wrapping_add(1)
}

fn launch_count() -> CubeCount {
    CubeCount::Static((N as u32).div_ceil(256), 1, 1)
}

fn read(client: &Client, handle: cubecl::server::Handle) -> Vec<u32> {
    u32::from_bytes(&client.read_one_unchecked(handle)).to_vec()
}

/// Read after write and write after write on one buffer, many times in one batch.
#[test]
fn an_in_place_chain_keeps_every_update() {
    let client = R::client(&Default::default());
    let start: Vec<u32> = (0..N as u32).collect();
    let values = client.create_from_slice(u32::as_bytes(&start));
    for _ in 0..300 {
        unsafe {
            step_kernel::launch_unchecked(
                &client,
                launch_count(),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(values.clone(), N),
            );
        }
    }
    let expected: Vec<u32> = start
        .iter()
        .map(|&x| (0..300).fold(x, |x, _| step(x)))
        .collect();
    assert_eq!(read(&client, values), expected);
}

/// Each stage reads the slice the previous one wrote, all slices of one allocation; the
/// stages before a slice was written must not see it, and stages on other slices don't
/// conflict.
#[test]
fn slices_of_one_buffer_chain_in_order() {
    let client = R::client(&Default::default());
    let stages = 8;
    let start: Vec<u32> = (0..N as u32).collect();
    let mut all = start.clone();
    all.resize(N * stages, 0);
    let buffer = client.create_from_slice(u32::as_bytes(&all));
    let bytes = (N * size_of::<u32>()) as u64;
    let total = bytes * stages as u64;
    let slice = |k: u64| {
        buffer
            .clone()
            .offset_start(k * bytes)
            .offset_end(total - (k + 1) * bytes)
    };
    for k in 0..stages as u64 - 1 {
        unsafe {
            step_into_kernel::launch_unchecked(
                &client,
                launch_count(),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(slice(k), N),
                BufferArg::from_raw_parts(slice(k + 1), N),
            );
        }
    }
    let got = read(&client, buffer);
    for k in 0..stages {
        let expected: Vec<u32> = start
            .iter()
            .map(|&x| (0..k).fold(x, |x, _| step(x)))
            .collect();
        assert_eq!(&got[k * N..(k + 1) * N], &expected[..], "stage {k}");
    }
}

/// A temporary freed after use hands its memory to the next one, so the next iteration's
/// write must wait for this iteration's read of the same bytes.
#[test]
fn reused_memory_waits_for_its_last_reader() {
    let client = R::client(&Default::default());
    let start: Vec<u32> = (0..N as u32).collect();
    let input = client.create_from_slice(u32::as_bytes(&start));
    let sum = client.create_from_slice(u32::as_bytes(&vec![0u32; N]));
    let rounds = 100;
    for _ in 0..rounds {
        let temporary = client.empty(N * size_of::<u32>());
        unsafe {
            step_into_kernel::launch_unchecked(
                &client,
                launch_count(),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(input.clone(), N),
                BufferArg::from_raw_parts(temporary.clone(), N),
            );
            accumulate_kernel::launch_unchecked(
                &client,
                launch_count(),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(temporary.clone(), N),
                BufferArg::from_raw_parts(sum.clone(), N),
            );
            // Overwrite the temporary right after it was read, so a missed
            // write-after-read hazard corrupts the sum above.
            step_into_kernel::launch_unchecked(
                &client,
                launch_count(),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(sum.clone(), N),
                BufferArg::from_raw_parts(temporary, N),
            );
        }
    }
    let expected: Vec<u32> = start
        .iter()
        .map(|&x| step(x).wrapping_mul(rounds))
        .collect();
    assert_eq!(read(&client, sum), expected);
}
