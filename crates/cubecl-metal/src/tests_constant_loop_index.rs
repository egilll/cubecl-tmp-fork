//! A mutable index initialized from a constant and updated in a loop, used
//! to address shared memory: the shape of a tree reduction whose width is
//! known at compile time.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

const UNITS: usize = 64;

#[cube(launch_unchecked)]
fn tree_sum(input: &[f32], output: &mut [f32]) {
    let mut scratch = Shared::<[f32]>::new_slice(UNITS);
    let unit = UNIT_POS as usize;
    scratch[unit] = input[unit];
    sync_cube();
    let mut half = UNITS / 2;
    while half > 0 {
        if unit < half {
            scratch[unit] = scratch[unit] + scratch[unit + half];
        }
        sync_cube();
        half /= 2;
    }
    if unit == 0 {
        output[0] = scratch[0];
    }
}

#[test]
fn a_constant_initialized_loop_index_addresses_shared_memory() {
    let client = R::client(&Default::default());
    let input: Vec<f32> = (0..UNITS).map(|i| i as f32).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(core::mem::size_of::<f32>());
    unsafe {
        tree_sum::launch_unchecked(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(UNITS as u32),
            BufferArg::from_raw_parts(input_handle, UNITS),
            BufferArg::from_raw_parts(output_handle.clone(), 1),
        );
    }
    let bytes = client.read_one_unchecked(output_handle);
    assert_eq!(f32::from_bytes(&bytes)[0], (UNITS * (UNITS - 1) / 2) as f32);
}

#[cube(launch_unchecked)]
fn halving_literal(output: &mut [u32]) {
    let mut half = 32usize;
    let mut steps = 0u32;
    while half > 0 {
        steps += 1;
        half /= 2;
    }
    output[0] = steps;
}

#[cube(launch_unchecked)]
fn halving_from_runtime(output: &mut [u32]) {
    let mut half = CUBE_DIM as usize / 2;
    let mut steps = 0u32;
    while half > 0 {
        steps += 1;
        half /= 2;
    }
    output[0] = steps;
}

fn halving(launch: impl Fn(&cubecl_core::prelude::Client, cubecl_core::server::Handle)) -> u32 {
    let client = R::client(&Default::default());
    let output = client.empty(4);
    launch(&client, output.clone());
    u32::from_bytes(&client.read_one_unchecked(output))[0]
}

#[test]
fn a_literal_initialized_loop_counter_halves() {
    let steps = halving(|client, output| unsafe {
        halving_literal::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(output, 1));
    });
    assert_eq!(steps, 6);
}

#[test]
fn a_runtime_initialized_loop_counter_halves() {
    let steps = halving(|client, output| unsafe {
        halving_from_runtime::launch_unchecked(client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(64), BufferArg::from_raw_parts(output, 1));
    });
    assert_eq!(steps, 6);
}
