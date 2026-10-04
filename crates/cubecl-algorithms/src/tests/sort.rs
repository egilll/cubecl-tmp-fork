use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::sort::sort_pairs;

fn check(client: &Client, keys: Vec<u32>) {
    let n = keys.len();
    let values: Vec<u32> = (0..n as u32).collect();
    let key_buffer = Buffer::create(client, &keys);
    let value_buffer = Buffer::create(client, &values);
    sort_pairs(client, &key_buffer, &value_buffer);
    let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(values).collect();
    // Stable: equal keys keep their original order, which `sort_by_key` keeps too.
    expected.sort_by_key(|(key, _)| *key);
    let got_keys = key_buffer.read(client).unwrap();
    let got_values = value_buffer.read(client).unwrap();
    for (i, (key, value)) in expected.into_iter().enumerate() {
        assert_eq!(
            (got_keys[i], got_values[i]),
            (key, value),
            "n {n}, index {i}"
        );
    }
}

pub fn test_sort_pairs<R: Runtime>(client: Client) {
    for n in [0usize, 1, 2, 255, 256, 257, 10_007, 300_000] {
        let keys: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2654435761).rotate_left(7))
            .collect();
        check(&client, keys);
    }
    // All equal, already sorted, reversed, and few distinct keys.
    check(&client, vec![42; 5000]);
    check(&client, (0..5000).collect());
    check(&client, (0..5000).rev().collect());
    check(&client, (0..5000u32).map(|i| i % 3).collect());
    check(&client, vec![u32::MAX, 0, u32::MAX - 1, 1]);
}

#[macro_export]
macro_rules! testgen_sort {
    () => {
        mod sort {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_sort_pairs() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::sort::test_sort_pairs::<TestRuntime>(client);
            }
        }
    };
}
