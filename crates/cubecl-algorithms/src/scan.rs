//! Prefix sums, stream compaction and an indirect-dispatch helper.
//!
//! Scans are reduce-then-scan: every cube reduces its tile, the tile sums are
//! scanned (recursively), then every cube scans its tile from its offset.
//! Unlike a single-pass decoupled look-back scan, no cube ever waits on
//! another, so it is correct on GPUs that give no forward-progress guarantee
//! between cubes (Apple GPUs, WebGPU).

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_runtime::server::CubeCountSelection;

/// Units per cube.
pub(crate) const UNITS: usize = 256;
/// Elements each unit scans.
const ITEMS: usize = 4;
/// Elements per cube.
pub(crate) const TILE: usize = UNITS * ITEMS;

/// The inclusive scan of `value` across the cube's units, in shared memory.
/// Every unit must call it.
#[cube]
pub(crate) fn cube_inclusive_scan<T: Numeric>(value: T, shared: &mut [T]) -> T {
    let unit = UNIT_POS as usize;
    shared[unit] = value;
    sync_cube();
    #[unroll]
    for step in 0..8u32 {
        let offset = comptime!(1usize << step);
        let add = select(
            unit >= offset,
            shared[unit.saturating_sub(offset)],
            T::from_int(0),
        );
        sync_cube();
        shared[unit] += add;
        sync_cube();
    }
    shared[unit]
}

/// The sum of `value` across the cube's units. Every unit must call it.
#[cube]
pub(crate) fn cube_sum<T: Numeric>(value: T, shared: &mut [T]) -> T {
    let total = cube_inclusive_scan::<T>(value, shared);
    let last = shared[UNITS - 1];
    sync_cube();
    let _ = total;
    last
}

#[cube(launch)]
fn tile_sums<T: Numeric>(input: &[T], sums: &mut [T], tiles: u32) {
    let tile = CUBE_POS;
    if tile < tiles as usize {
        let mut shared = Shared::<[T]>::new_slice(UNITS);
        let base = tile * TILE + UNIT_POS as usize * ITEMS;
        let mut total = T::from_int(0);
        #[unroll]
        for k in 0..ITEMS {
            if base + k < input.len() {
                total += input[base + k];
            }
        }
        let sum = cube_sum::<T>(total, &mut shared);
        if UNIT_POS == 0 {
            sums[tile] = sum;
        }
    }
}

#[cube(launch)]
fn tile_scan<T: Numeric>(
    input: &[T],
    offsets: &[T],
    output: &mut [T],
    tiles: u32,
    #[comptime] inclusive: bool,
) {
    let tile = CUBE_POS;
    if tile < tiles as usize {
        let mut shared = Shared::<[T]>::new_slice(UNITS);
        let base = tile * TILE + UNIT_POS as usize * ITEMS;
        let mut items = Array::<T>::new(ITEMS);
        let mut total = T::from_int(0);
        #[unroll]
        for k in 0..ITEMS {
            let value = select(base + k < input.len(), input[base + k], T::from_int(0));
            items[k] = value;
            total += value;
        }
        let before = cube_inclusive_scan::<T>(total, &mut shared) - total;
        let mut running = offsets[tile] + before;
        #[unroll]
        for k in 0..ITEMS {
            if comptime!(inclusive) {
                running += items[k];
            }
            if base + k < output.len() {
                output[base + k] = running;
            }
            if comptime!(!inclusive) {
                running += items[k];
            }
        }
    }
}

fn cube_count(client: &Client, cubes: usize) -> CubeCount {
    CubeCountSelection::new(client, cubes as u32).cube_count()
}

/// Write the prefix sums of `input` to `output` (same length): inclusive
/// (`output[i] = input[0] + … + input[i]`) or exclusive (`… + input[i - 1]`,
/// starting at zero).
pub fn scan<T: Numeric + CubeElement>(
    client: &Client,
    input: &Buffer<T>,
    output: &Buffer<T>,
    inclusive: bool,
) {
    assert_eq!(
        input.len(),
        output.len(),
        "a scan's output matches its input"
    );
    let n = input.len();
    if n == 0 {
        return;
    }
    let tiles = n.div_ceil(TILE);
    let offsets = if tiles == 1 {
        Buffer::create(client, &[T::from_int(0)])
    } else {
        let sums = Buffer::<T>::empty(client, tiles);
        tile_sums::launch::<T>(
            client,
            cube_count(client, tiles),
            CubeDim::new_1d(UNITS as u32),
            input.into(),
            (&sums).into(),
            tiles as u32,
        );
        let offsets = Buffer::<T>::empty(client, tiles);
        scan(client, &sums, &offsets, false);
        offsets
    };
    tile_scan::launch::<T>(
        client,
        cube_count(client, tiles),
        CubeDim::new_1d(UNITS as u32),
        input.into(),
        (&offsets).into(),
        output.into(),
        tiles as u32,
        inclusive,
    );
}

#[cube(launch)]
fn scatter_flagged<T: Numeric>(
    values: &[T],
    flags: &[u32],
    positions: &[u32],
    output: &mut [T],
    count: &mut [u32],
) {
    let i = ABSOLUTE_POS;
    let n = values.len();
    if i < n {
        if flags[i] != 0 {
            output[positions[i] as usize] = values[i];
        }
        if i == n - 1 {
            count[0] = positions[i] + select(flags[i] != 0, 1u32, 0u32);
        }
    }
}

/// Keep the `values` whose `flags` are non-zero, in order, at the front of
/// the returned buffer (as long as `values`), and the kept count in the
/// returned one-element buffer. The count stays on the device: feed it to
/// [`dispatch_args`] to size the next launch without a round trip.
pub fn compact<T: Numeric + CubeElement>(
    client: &Client,
    values: &Buffer<T>,
    flags: &Buffer<u32>,
) -> (Buffer<T>, Buffer<u32>) {
    assert_eq!(values.len(), flags.len(), "one flag per value");
    let n = values.len();
    let output = Buffer::<T>::empty(client, n);
    let count = Buffer::create(client, &[0u32]);
    if n == 0 {
        return (output, count);
    }
    let positions = Buffer::<u32>::empty(client, n);
    scan(client, flags, &positions, false);
    scatter_flagged::launch::<T>(
        client,
        cubecl_core::calculate_cube_count_elemwise(client, n, CubeDim::new_1d(UNITS as u32)),
        CubeDim::new_1d(UNITS as u32),
        values.into(),
        flags.into(),
        (&positions).into(),
        (&output).into(),
        (&count).into(),
    );
    (output, count)
}

#[cube(launch)]
fn dispatch_from_count(count: &[u32], args: &mut [u32], units_per_cube: u32, capacity: u32) {
    if UNIT_POS == 0 {
        let units = select(count[0] < capacity, count[0], capacity);
        args[0] = units.div_ceil(units_per_cube);
        args[1] = 1;
        args[2] = 1;
    }
}

/// The `[x, 1, 1]` cube count that covers a device-side `count` of units
/// with `units_per_cube` units each, clamped to `capacity` units, for
/// [`CubeCount::Dynamic`]: a launch sized by work the GPU produced, with no
/// round trip to the host.
pub fn dispatch_args(
    client: &Client,
    count: &Buffer<u32>,
    units_per_cube: u32,
    capacity: u32,
) -> Buffer<u32> {
    let args = Buffer::<u32>::empty(client, 3);
    dispatch_from_count::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        count.into(),
        (&args).into(),
        units_per_cube,
        capacity,
    );
    args
}
