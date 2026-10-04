//! Stable LSD radix sort of `u32` keys with `u32` payloads.
//!
//! Eight passes of four bits. Each pass counts every digit per tile of 256
//! keys, scans the counts digit-major across tiles (so all zeros come first,
//! then all ones…, each in tile order), and scatters every key to its
//! digit's offset plus its rank among equal digits in its tile. Ranking by
//! a cube-wide scan keeps equal keys in order, which is what makes the sort
//! stable, and no cube waits on another.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_runtime::server::CubeCountSelection;

use crate::scan::{UNITS, cube_inclusive_scan, cube_sum, scan};

const RADIX_BITS: u32 = 4;
const DIGITS: usize = 1 << RADIX_BITS;

/// The digit of key `i`, or `DIGITS` (counted nowhere) past the end.
#[cube]
fn digit_of(keys: &[u32], i: usize, #[comptime] shift: u32) -> u32 {
    select(
        i < keys.len(),
        (keys[i.min(keys.len() - 1)] >> shift) & (DIGITS as u32 - 1),
        DIGITS as u32,
    )
}

#[cube(launch)]
fn count_digits(keys: &[u32], counts: &mut [u32], tiles: u32, #[comptime] shift: u32) {
    let tile = CUBE_POS;
    if tile < tiles as usize {
        let mut shared = Shared::<[u32]>::new_slice(UNITS);
        let digit = digit_of(keys, tile * UNITS + UNIT_POS as usize, shift);
        #[unroll]
        for d in 0..DIGITS {
            let count = cube_sum::<u32>(select(digit == d as u32, 1u32, 0u32), &mut shared);
            if UNIT_POS == 0 {
                counts[d * tiles as usize + tile] = count;
            }
        }
    }
}

#[cube(launch)]
fn scatter_digits(
    keys: &[u32],
    values: &[u32],
    offsets: &[u32],
    keys_out: &mut [u32],
    values_out: &mut [u32],
    tiles: u32,
    #[comptime] shift: u32,
) {
    let tile = CUBE_POS;
    if tile < tiles as usize {
        let mut shared = Shared::<[u32]>::new_slice(UNITS);
        let i = tile * UNITS + UNIT_POS as usize;
        let digit = digit_of(keys, i, shift);
        let mut rank = 0u32;
        #[unroll]
        for d in 0..DIGITS {
            let mine = select(digit == d as u32, 1u32, 0u32);
            let before = cube_inclusive_scan::<u32>(mine, &mut shared) - mine;
            rank = select(digit == d as u32, before, rank);
        }
        if i < keys.len() {
            let dest = offsets[digit as usize * tiles as usize + tile] + rank;
            keys_out[dest as usize] = keys[i];
            values_out[dest as usize] = values[i];
        }
    }
}

/// Sort `keys` ascending, moving each `values` entry with its key; equal
/// keys keep their order. Both buffers are rewritten in place.
pub fn sort_pairs(client: &Client, keys: &Buffer<u32>, values: &Buffer<u32>) {
    assert_eq!(keys.len(), values.len(), "one value per key");
    let n = keys.len();
    if n <= 1 {
        return;
    }
    let tiles = n.div_ceil(UNITS);
    let count = CubeCountSelection::new(client, tiles as u32).cube_count();
    let dim = CubeDim::new_1d(UNITS as u32);
    let counts = Buffer::<u32>::empty(client, DIGITS * tiles);
    let offsets = Buffer::<u32>::empty(client, DIGITS * tiles);
    let mut from = (keys.clone(), values.clone());
    let mut to = (
        Buffer::<u32>::empty(client, n),
        Buffer::<u32>::empty(client, n),
    );
    for pass in 0..(32 / RADIX_BITS) {
        let shift = pass * RADIX_BITS;
        count_digits::launch(
            client,
            count.clone(),
            dim,
            (&from.0).into(),
            (&counts).into(),
            tiles as u32,
            shift,
        );
        scan(client, &counts, &offsets, false);
        scatter_digits::launch(
            client,
            count.clone(),
            dim,
            (&from.0).into(),
            (&from.1).into(),
            (&offsets).into(),
            (&to.0).into(),
            (&to.1).into(),
            tiles as u32,
            shift,
        );
        core::mem::swap(&mut from, &mut to);
    }
    // Eight passes, an even number, end in the caller's buffers.
}
