use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::reduce::segmented_sum;

pub fn test_segmented_sum<R: Runtime>(client: Client) {
    // Empty, single-element, tile-sized and long segments, back to back.
    let lengths = [0usize, 1, 3, 0, 256, 257, 1000, 7, 0, 70_001];
    let mut offsets = vec![0u32];
    for len in lengths {
        offsets.push(offsets.last().unwrap() + len as u32);
    }
    let n = *offsets.last().unwrap() as usize;
    let values: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(7919) % 1013).collect();
    let sums = segmented_sum(
        &client,
        &Buffer::create(&client, &values),
        &Buffer::create(&client, &offsets),
    )
    .read(&client)
    .unwrap();
    for (s, window) in offsets.windows(2).enumerate() {
        let expected: u32 = values[window[0] as usize..window[1] as usize].iter().sum();
        assert_eq!(sums[s], expected, "segment {s}");
    }
}

#[macro_export]
macro_rules! testgen_reduce {
    () => {
        mod reduce {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_segmented_sum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::reduce::test_segmented_sum::<TestRuntime>(client);
            }
        }
    };
}
