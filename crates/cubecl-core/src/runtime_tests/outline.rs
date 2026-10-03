use crate::prelude::*;
use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

// `#[cube(outline)]` functions, called from several sites, with runtime and
// constant arguments, from each other, and with loops and branches inside.
// Whether a target keeps the calls or inlines them, the results must match
// plain Rust.

#[cube(outline)]
fn poly(x: f32, a: f32, b: f32) -> f32 {
    (x * a + b) * x - a
}

#[cube(outline)]
fn pick(flag: bool, x: f32, y: f32) -> f32 {
    if flag {
        poly(x, y, 1.0)
    } else {
        poly(y, x, 2.0)
    }
}

#[cube(outline)]
fn collatz_steps(start: u32, limit: u32) -> u32 {
    let mut n = start;
    let mut steps = 0u32;
    while n != 1 && steps < limit {
        if n % 2 == 0 {
            n /= 2;
        } else {
            n = 3 * n + 1;
        }
        steps += 1;
    }
    steps
}

#[cube(launch)]
pub fn kernel_outlined_calls(input: &[f32], output: &mut [f32], steps: &mut [u32]) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        let mut acc = 0.0f32;
        #[unroll]
        for k in 0..4u32 {
            acc += poly(x, acc + f32::cast_from(k), 0.5);
        }
        acc += pick(i % 2 == 0, x, acc * 0.001);
        acc += pick(i % 3 == 0, acc * 0.001, x);
        output[i] = acc;
        steps[i] = collatz_steps(i as u32 + 1, 200) + collatz_steps(i as u32 + 7, 200);
    }
}

fn poly_ref(x: f32, a: f32, b: f32) -> f32 {
    (x * a + b) * x - a
}

fn pick_ref(flag: bool, x: f32, y: f32) -> f32 {
    match flag {
        true => poly_ref(x, y, 1.0),
        false => poly_ref(y, x, 2.0),
    }
}

fn collatz_ref(start: u32, limit: u32) -> u32 {
    let (mut n, mut steps) = (start, 0);
    while n != 1 && steps < limit {
        n = if n.is_multiple_of(2) {
            n / 2
        } else {
            3 * n + 1
        };
        steps += 1;
    }
    steps
}

pub fn test_outlined_calls<R: Runtime>(client: Client) {
    let n = 96usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(n * core::mem::size_of::<f32>());
    let steps_handle = client.empty(n * core::mem::size_of::<u32>());

    kernel_outlined_calls::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(n as u32),
        unsafe { BufferArg::from_raw_parts(input_handle, n) },
        unsafe { BufferArg::from_raw_parts(output_handle.clone(), n) },
        unsafe { BufferArg::from_raw_parts(steps_handle.clone(), n) },
    );

    let output = client.read_one_unchecked(output_handle);
    let output = f32::from_bytes(&output);
    let steps = client.read_one_unchecked(steps_handle);
    let steps = u32::from_bytes(&steps);

    for i in 0..n {
        let x = input[i];
        let mut acc = 0.0f32;
        for k in 0..4 {
            acc += poly_ref(x, acc + k as f32, 0.5);
        }
        acc += pick_ref(i % 2 == 0, x, acc * 0.001);
        acc += pick_ref(i % 3 == 0, acc * 0.001, x);
        let tolerance = 1e-3 * acc.abs().max(1.0);
        assert!(
            (output[i] - acc).abs() <= tolerance,
            "output[{i}] = {}, expected {acc}",
            output[i]
        );

        let expected = collatz_ref(i as u32 + 1, 200) + collatz_ref(i as u32 + 7, 200);
        assert_eq!(steps[i], expected, "steps[{i}]");
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_outline {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_outlined_calls() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::outline::test_outlined_calls::<TestRuntime>(client);
        }
    };
}
