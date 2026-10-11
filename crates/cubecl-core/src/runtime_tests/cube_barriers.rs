//! Cube barriers in the shapes kernels use them: inside loops, with values and variables each
//! unit holds across them, with units that leave early, and with cubes larger than a CPU has
//! cores.

use alloc::{vec, vec::Vec};
use cubecl_runtime::runtime::Runtime;

use crate::prelude::*;
use crate::{self as cubecl};

/// The largest power of two up to `limit` units a cube of this client may have.
fn cube_units(client: &Client, limit: u32) -> u32 {
    let max = client.properties().hardware.max_units_per_cube.min(limit);
    1 << (u32::BITS - 1 - max.leading_zeros())
}

#[cube(launch)]
fn kernel_tree_sum(input: &[u32], out: &mut [u32], #[comptime] units: usize) {
    let mut partial = Shared::<[u32]>::new_slice(units);
    partial[UNIT_POS] = input[ABSOLUTE_POS];
    sync_cube();
    let mut stride = units / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            partial[UNIT_POS] = partial[UNIT_POS] + partial[UNIT_POS + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if UNIT_POS == 0 {
        out[CUBE_POS] = partial[0];
    }
}

/// Each cube sums its units' inputs by halving, with a barrier on every step.
pub fn test_tree_sum<R: Runtime>(client: Client) {
    let units = cube_units(&client, 256);
    let cubes = 3u32;
    let input: Vec<u32> = (0..units * cubes).map(|i| i * 7 % 13).collect();
    let input_handle = client.create_from_slice(u32::as_bytes(&input));
    let out = client.empty(cubes as usize * core::mem::size_of::<u32>());

    kernel_tree_sum::launch(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(input_handle, input.len()) },
        unsafe { BufferArg::from_raw_parts(out.clone(), cubes as usize) },
        units as usize,
    );

    let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    let expected: Vec<u32> = input
        .chunks(units as usize)
        .map(|cube| cube.iter().sum())
        .collect();
    assert_eq!(actual, expected);
}

#[cube(launch)]
fn kernel_held_across_barriers(input: &[u32], out: &mut [u32], #[comptime] units: usize) {
    let mut exchange = Shared::<[u32]>::new_slice(units);
    // Loaded, so it cannot be recomputed after the barriers.
    let own = input[ABSOLUTE_POS] * 3 + 1;
    let mut local = Array::<u32>::new(4usize);
    #[unroll]
    for i in 0..4usize {
        local[i] = own + i as u32;
    }
    let mut total = own;
    for round in 0..3u32 {
        exchange[UNIT_POS] = total + round;
        sync_cube();
        total += exchange[(UNIT_POS + 1) % units];
        local[(round as usize) % 4] += total;
        sync_cube();
    }
    out[ABSOLUTE_POS] = own + total + local[0] + local[1] + local[2] + local[3];
}

/// A unit's loaded value, its local array and its running total survive barriers inside a loop
/// while every other unit of its cube runs.
pub fn test_held_across_barriers<R: Runtime>(client: Client) {
    let units = cube_units(&client, 64);
    let cubes = 2u32;
    let n = (units * cubes) as usize;
    let input: Vec<u32> = (0..n as u32).map(|i| i * 5 % 11).collect();
    let input_handle = client.create_from_slice(u32::as_bytes(&input));
    let out = client.empty(n * core::mem::size_of::<u32>());

    kernel_held_across_barriers::launch(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(input_handle, n) },
        unsafe { BufferArg::from_raw_parts(out.clone(), n) },
        units as usize,
    );

    let units = units as usize;
    let mut expected = vec![0u32; n];
    for cube in 0..cubes as usize {
        let own: Vec<u32> = (0..units)
            .map(|unit| input[cube * units + unit] * 3 + 1)
            .collect();
        let mut local: Vec<[u32; 4]> = own
            .iter()
            .map(|&own| [own, own + 1, own + 2, own + 3])
            .collect();
        let mut total = own.clone();
        for round in 0..3u32 {
            let exchange: Vec<u32> = total.iter().map(|t| t + round).collect();
            for unit in 0..units {
                total[unit] += exchange[(unit + 1) % units];
                local[unit][round as usize % 4] += total[unit];
            }
        }
        for unit in 0..units {
            expected[cube * units + unit] =
                own[unit] + total[unit] + local[unit].iter().sum::<u32>();
        }
    }
    let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    assert_eq!(actual, expected);
}

#[cube(launch)]
fn kernel_cubes_finish_apart(out: &mut [u32], #[comptime] units: usize) {
    let mut exchange = Shared::<[u32]>::new_slice(units);
    if CUBE_POS == 0 {
        terminate!();
    }
    exchange[UNIT_POS] = UNIT_POS as u32;
    sync_cube();
    if UNIT_POS % 2 == 1 {
        terminate!();
    }
    out[ABSOLUTE_POS] = exchange[(UNIT_POS + 1) % units] + 1;
    sync_cube();
}

/// A cube whose units all leave before a barrier leaves the other cubes running, and units that
/// leave between barriers leave the rest of their cube running.
pub fn test_cubes_finish_apart<R: Runtime>(client: Client) {
    let units = cube_units(&client, 32);
    let cubes = 3u32;
    let n = (units * cubes) as usize;
    let out = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));

    kernel_cubes_finish_apart::launch(
        &client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(out.clone(), n) },
        units as usize,
    );

    let units = units as usize;
    let expected: Vec<u32> = (0..n)
        .map(|i| {
            let (cube, unit) = (i / units, i % units);
            if cube == 0 || unit % 2 == 1 {
                0
            } else {
                ((unit + 1) % units) as u32 + 1
            }
        })
        .collect();
    let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    assert_eq!(actual, expected);
}

#[cube(launch)]
fn kernel_flag_across_barriers(out: &mut [u32], #[comptime] units: usize) {
    let mut exchange = Shared::<[u32]>::new_slice(units);
    let mut counted = true;
    exchange[UNIT_POS] = 1u32;
    sync_cube();
    if exchange[(UNIT_POS + 1) % units] == 1u32 && UNIT_POS % 3 == 0 {
        counted = false;
    }
    sync_cube();
    exchange[UNIT_POS] = 2u32;
    sync_cube();
    out[ABSOLUTE_POS] = if counted { exchange[UNIT_POS] } else { 0u32 };
}

/// A variable initialized before a barrier, set between two and read after them, is each unit's
/// own.
pub fn test_flag_across_barriers<R: Runtime>(client: Client) {
    let units = cube_units(&client, 16);
    let n = units as usize;
    let out = client.empty(n * core::mem::size_of::<u32>());

    kernel_flag_across_barriers::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(out.clone(), n) },
        units as usize,
    );

    let expected: Vec<u32> = (0..n)
        .map(|unit| if unit % 3 == 0 { 0 } else { 2 })
        .collect();
    let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    assert_eq!(actual, expected);
}

#[cube(launch)]
fn kernel_positions(out: &mut [u32], #[comptime] barrier: bool) {
    let at = ABSOLUTE_POS * 12;
    out[at] = UNIT_POS_X as u32;
    out[at + 1] = UNIT_POS_Y as u32;
    out[at + 2] = UNIT_POS_Z as u32;
    out[at + 3] = CUBE_POS_X as u32;
    out[at + 4] = CUBE_POS_Y as u32;
    out[at + 5] = CUBE_POS_Z as u32;
    if barrier {
        sync_cube();
    }
    out[at + 6] = UNIT_POS as u32;
    out[at + 7] = CUBE_POS as u32;
    out[at + 8] = ABSOLUTE_POS_X as u32;
    out[at + 9] = ABSOLUTE_POS_Y as u32;
    out[at + 10] = ABSOLUTE_POS_Z as u32;
    out[at + 11] = CUBE_COUNT as u32;
}

/// Every position builtin of three-dimensional cubes and counts, with and without a barrier
/// between reading them.
pub fn test_positions<R: Runtime>(client: Client) {
    let units = cube_units(&client, 64);
    let dim = match units {
        64.. => (4, 4, 4),
        8.. => (2, 2, 2),
        2.. => (2, 1, 1),
        _ => (1, 1, 1),
    };
    let count = (3, 2, 2);
    let units = dim.0 * dim.1 * dim.2;
    let cubes = count.0 * count.1 * count.2;
    let n = (units * cubes * 12) as usize;

    let mut expected = vec![0u32; n];
    for cz in 0..count.2 {
        for cy in 0..count.1 {
            for cx in 0..count.0 {
                for uz in 0..dim.2 {
                    for uy in 0..dim.1 {
                        for ux in 0..dim.0 {
                            let cube = cx + cy * count.0 + cz * count.0 * count.1;
                            let unit = ux + uy * dim.0 + uz * dim.0 * dim.1;
                            let at = ((cube * units + unit) * 12) as usize;
                            expected[at..at + 12].copy_from_slice(&[
                                ux,
                                uy,
                                uz,
                                cx,
                                cy,
                                cz,
                                unit,
                                cube,
                                cx * dim.0 + ux,
                                cy * dim.1 + uy,
                                cz * dim.2 + uz,
                                cubes,
                            ]);
                        }
                    }
                }
            }
        }
    }

    for barrier in [false, true] {
        let out = client.empty(n * core::mem::size_of::<u32>());
        kernel_positions::launch(
            &client,
            CubeCount::Static(count.0, count.1, count.2),
            CubeDim::new_3d(dim.0, dim.1, dim.2),
            unsafe { BufferArg::from_raw_parts(out.clone(), n) },
            barrier,
        );
        let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
        assert_eq!(actual, expected, "with a barrier: {barrier}");
    }
}

#[cube(launch)]
fn kernel_one_wide_cube(out: &mut [u32]) {
    out[ABSOLUTE_POS] = ABSOLUTE_POS as u32 * 2 + 1;
}

/// A single cube of as many units as the client allows, with no barrier: every unit runs once.
pub fn test_one_wide_cube<R: Runtime>(client: Client) {
    let units = cube_units(&client, 1024);
    let n = units as usize;
    let out = client.empty(n * core::mem::size_of::<u32>());

    kernel_one_wide_cube::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(units),
        unsafe { BufferArg::from_raw_parts(out.clone(), n) },
    );

    let expected: Vec<u32> = (0..units).map(|i| i * 2 + 1).collect();
    let actual = u32::from_bytes(&client.read_one_unchecked(out)).to_vec();
    assert_eq!(actual, expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cube_barriers {
    () => {
        mod cube_barriers {
            use super::*;

            #[$crate::runtime_tests::test_log::test]
            fn test_tree_sum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_tree_sum::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_held_across_barriers() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_held_across_barriers::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_cubes_finish_apart() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_cubes_finish_apart::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_flag_across_barriers() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_flag_across_barriers::<TestRuntime>(
                    client,
                );
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_positions() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_positions::<TestRuntime>(client);
            }

            #[$crate::runtime_tests::test_log::test]
            fn test_one_wide_cube() {
                let client = TestRuntime::client(&Default::default());
                cubecl_core::runtime_tests::cube_barriers::test_one_wide_cube::<TestRuntime>(
                    client,
                );
            }
        }
    };
}
