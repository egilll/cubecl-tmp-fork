//! Segmented reduction: the sum of every segment of a flat array.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_runtime::server::CubeCountSelection;

use crate::scan::{UNITS, cube_sum};

#[cube(launch)]
fn sum_segments<T: Numeric>(values: &[T], offsets: &[u32], sums: &mut [T]) {
    let segment = CUBE_POS;
    if segment < sums.len() {
        let mut shared = Shared::<[T]>::new_slice(UNITS);
        let end = offsets[segment + 1] as usize;
        let mut i = offsets[segment] as usize + UNIT_POS as usize;
        let mut total = T::from_int(0);
        while i < end {
            total += values[i];
            i += UNITS;
        }
        let sum = cube_sum::<T>(total, &mut shared);
        if UNIT_POS == 0 {
            sums[segment] = sum;
        }
    }
}

/// The sum of each segment `values[offsets[s]..offsets[s + 1]]`, one per
/// segment (`offsets.len() - 1` of them). Offsets must not decrease; an
/// empty segment sums to zero. One cube per segment, so it suits segments of
/// a few hundred elements or more; many tiny segments waste units.
pub fn segmented_sum<T: Numeric + CubeElement>(
    client: &Client,
    values: &Buffer<T>,
    offsets: &Buffer<u32>,
) -> Buffer<T> {
    assert!(
        !offsets.is_empty(),
        "offsets start with the first segment's"
    );
    let segments = offsets.len() - 1;
    let sums = Buffer::<T>::empty(client, segments.max(1));
    if segments > 0 {
        sum_segments::launch::<T>(
            client,
            CubeCountSelection::new(client, segments as u32).cube_count(),
            CubeDim::new_1d(UNITS as u32),
            values.into(),
            offsets.into(),
            (&sums).into(),
        );
    }
    sums
}
