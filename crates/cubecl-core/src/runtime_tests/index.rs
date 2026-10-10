use crate::{self as cubecl};
use cubecl_runtime::runtime::Runtime;

use alloc::vec;
use cubecl::prelude::*;

#[cube(launch)]
fn shuffle_kernel(orders: &mut [u32]) {
    if UNIT_POS < 2 {
        let mut order = Array::new(2usize);
        order[0] = 0;
        order[1] = 1;

        let i = ABSOLUTE_POS & 1;
        let tmp = order[i];
        order[i] = 9;
        order[i ^ 1] = tmp;

        let base = ABSOLUTE_POS * 2;
        orders[base] = order[0];
        orders[base + 1] = order[1];
    }
}

#[cube(launch)]
fn destructurable_kernel(orders: &mut [u32]) {
    if UNIT_POS == 0 {
        let mut order = Array::new(4usize);
        order[0] = orders[0];
        order[1] = orders[1];
        order[2] = orders[2];
        order[3] = orders[3];
        let len = order.len().comptime();

        #[unroll]
        for i in 0..len {
            order[i] += order[len - i - 1] + i as u32;
        }

        #[unroll]
        for i in 0..len {
            // ABSOLUTE_POS prevents constant inlining and keeps index dynamic
            orders[ABSOLUTE_POS + i + 4] = order[i];
        }
    }
}

#[cube(launch)]
fn literal_kernel(values: &mut [u32]) {
    if UNIT_POS == 0 {
        // Runtime elements, so the literal is a local array rather than a constant.
        let order = [values[0], values[1] + 1, values[2] * 2];
        // Read before the loop overwrites it.
        let rotation = values[3] as usize;
        for i in 0..3usize {
            // A dynamic index into the literal.
            values[ABSOLUTE_POS + i + 3] = order[(i + rotation) % 3];
        }
    }
}

// Regression test for invalid `CopyTransform`
pub fn test_kernel_shuffle<R: Runtime>(client: Client) {
    let handle = client.empty(4 * size_of::<u32>());

    shuffle_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 4) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, [9, 0, 1, 9]);
}

// Test to ensure destructuring arrays doesn't break anything
pub fn test_kernel_destructurable<R: Runtime>(client: Client) {
    let data = vec![1, 2, 3, 4, 10, 10, 10, 10];
    let handle = client.create_from_slice(u32::as_bytes(&data));

    destructurable_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 8) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, [1, 2, 3, 4, 5, 6, 11, 12]);
}

// Array literals with runtime elements, read at runtime indices
pub fn test_kernel_literal<R: Runtime>(client: Client) {
    let data = vec![4, 5, 6, 1, 0, 0];
    let handle = client.create_from_slice(u32::as_bytes(&data));

    literal_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 6) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, [4, 5, 6, 6, 12, 4]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_index {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_shuffle() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_shuffle::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_literal() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_literal::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_destructurable() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::index::test_kernel_destructurable::<TestRuntime>(client);
        }
    };
}
