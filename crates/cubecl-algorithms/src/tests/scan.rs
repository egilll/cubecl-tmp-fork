use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::scan::{compact, dispatch_args, scan};

/// Sizes that exercise empty, single, partial-tile, exact-tile and
/// recursive (more than 1024 tiles) scans.
const SIZES: &[usize] = &[0, 1, 7, 1023, 1024, 1025, 4099, 1 << 20, (1 << 20) + 4093];

pub fn test_scan<R: Runtime>(client: Client) {
    for &n in SIZES {
        let input: Vec<u32> = (0..n as u32)
            .map(|i| i.wrapping_mul(2654435761) % 7)
            .collect();
        let input_buffer = Buffer::create(&client, &input);
        for inclusive in [false, true] {
            let output = Buffer::<u32>::empty(&client, n);
            scan(&client, &input_buffer, &output, inclusive);
            let got = output.read(&client).unwrap();
            let mut running = 0u32;
            for (i, &x) in input.iter().enumerate() {
                let expected = if inclusive { running + x } else { running };
                assert_eq!(got[i], expected, "n {n}, inclusive {inclusive}, index {i}");
                running += x;
            }
        }
    }
}

pub fn test_scan_f32<R: Runtime>(client: Client) {
    let input: Vec<f32> = (0..5000).map(|i| (i % 3) as f32 * 0.5).collect();
    let input_buffer = Buffer::create(&client, &input);
    let output = Buffer::<f32>::empty(&client, input.len());
    scan(&client, &input_buffer, &output, true);
    let got = output.read(&client).unwrap();
    let mut running = 0.0f32;
    for (i, &x) in input.iter().enumerate() {
        running += x;
        assert_eq!(got[i], running, "index {i}");
    }
}

pub fn test_compact<R: Runtime>(client: Client) {
    for &n in &[0usize, 1, 1000, 70_001] {
        let values: Vec<u32> = (0..n as u32).collect();
        let flags: Vec<u32> = values.iter().map(|v| u32::from(v % 3 == 1)).collect();
        let (kept, count) = compact(
            &client,
            &Buffer::create(&client, &values),
            &Buffer::create(&client, &flags),
        );
        let expected: Vec<u32> = values.iter().copied().filter(|v| v % 3 == 1).collect();
        assert_eq!(
            count.read(&client).unwrap(),
            vec![expected.len() as u32],
            "n {n}"
        );
        assert_eq!(
            &kept.read(&client).unwrap()[..expected.len()],
            &expected[..],
            "n {n}"
        );
    }
}

#[cube(launch)]
fn mark(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 1;
    }
}

/// A launch sized by a count the GPU wrote covers exactly that many cubes.
pub fn test_dispatch_args<R: Runtime>(client: Client) {
    let count = Buffer::create(&client, &[1000u32]);
    let args = dispatch_args(&client, &count, 64, 100_000);
    assert_eq!(args.read(&client).unwrap(), vec![16, 1, 1]);
    let output = Buffer::create(&client, &vec![0u32; 2048]);
    mark::launch(
        &client,
        CubeCount::Dynamic(args.handle().clone().binding()),
        CubeDim::new_1d(64),
        (&output).into(),
    );
    let marked = output
        .read(&client)
        .unwrap()
        .iter()
        .filter(|&&x| x == 1)
        .count();
    assert_eq!(marked, 16 * 64);

    // Clamped to the capacity.
    let args = dispatch_args(&client, &count, 64, 100);
    assert_eq!(args.read(&client).unwrap(), vec![2, 1, 1]);
}

#[macro_export]
macro_rules! testgen_scan {
    () => {
        mod scan {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_scan() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::scan::test_scan::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_scan_f32() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::scan::test_scan_f32::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_compact() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::scan::test_compact::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_dispatch_args() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::scan::test_dispatch_args::<TestRuntime>(client);
            }
        }
    };
}
