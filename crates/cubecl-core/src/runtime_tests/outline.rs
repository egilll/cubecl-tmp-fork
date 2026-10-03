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

/// Large enough to be kept as a call, and reads a builtin.
#[cube(outline)]
fn blend(x: f32, y: f32, t: f32) -> f32 {
    let unit = f32::cast_from(UNIT_POS) * 0.25;
    let a = x * (1.0 - t) + y * t;
    let b = (a * a + unit) / (1.0 + a * a);
    let c = b * 3.0 - a * 0.5 + t * t;
    let d = c * c * 0.125 + b * 0.75 - unit * 0.5;
    (d + a * 0.25) * (1.0 + t * 0.125) - c * 0.0625
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
        let t = f32::cast_from(i % 4) * 0.25;
        acc += blend(x, acc * 0.01, t);
        acc += blend(acc * 0.01, x, 1.0 - t);
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

fn blend_ref(unit_pos: usize, x: f32, y: f32, t: f32) -> f32 {
    let unit = unit_pos as f32 * 0.25;
    let a = x * (1.0 - t) + y * t;
    let b = (a * a + unit) / (1.0 + a * a);
    let c = b * 3.0 - a * 0.5 + t * t;
    let d = c * c * 0.125 + b * 0.75 - unit * 0.5;
    (d + a * 0.25) * (1.0 + t * 0.125) - c * 0.0625
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
        let t = (i % 4) as f32 * 0.25;
        acc += blend_ref(i, x, acc * 0.01, t);
        acc += blend_ref(i, acc * 0.01, x, 1.0 - t);
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
