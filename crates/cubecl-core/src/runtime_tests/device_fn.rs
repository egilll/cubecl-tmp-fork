use crate::prelude::*;
use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

// `#[cube(inline(never))]` functions, called from several sites, with runtime and
// constant arguments, from each other, and with loops and branches inside.
// Whether a target keeps the calls or inlines them, the results must match
// plain Rust.

#[cube(inline(never))]
fn poly(x: f32, a: f32, b: f32) -> f32 {
    (x * a + b) * x - a
}

#[cube(inline(never))]
fn pick(flag: bool, x: f32, y: f32) -> f32 {
    if flag {
        poly(x, y, 1.0)
    } else {
        poly(y, x, 2.0)
    }
}

#[cube(inline(never))]
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
#[cube(inline(never))]
fn blend(x: f32, y: f32, t: f32) -> f32 {
    let unit = f32::cast_from(UNIT_POS) * 0.25;
    let a = x * (1.0 - t) + y * t;
    let b = (a * a + unit) / (1.0 + a * a);
    let c = b * 3.0 - a * 0.5 + t * t;
    let d = c * c * 0.125 + b * 0.75 - unit * 0.5;
    (d + a * 0.25) * (1.0 + t * 0.125) - c * 0.0625
}

#[cube(launch)]
pub fn kernel_device_fn_calls(input: &[f32], output: &mut [f32], steps: &mut [u32]) {
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

#[derive(CubeType, Clone, Copy)]
struct Curve {
    scale: f32,
    shift: f32,
    #[cube(comptime)]
    steps: u32,
}

#[cube]
impl Curve {
    fn new(scale: f32, shift: f32, #[comptime] steps: u32) -> Curve {
        Curve {
            scale,
            shift,
            steps,
        }
    }

    #[cube(inline(never))]
    fn eval(&self, x: f32) -> f32 {
        let mut y = x * self.scale + self.shift;
        #[unroll]
        for _ in 0..self.steps {
            y = y * 0.5 + x * self.scale * 0.25 - self.shift * 0.125;
        }
        y
    }
}

#[cube]
trait Response: CubeType {
    fn respond(&self, x: f32) -> f32;
}

#[cube]
impl Response for Curve {
    #[cube(inline(never))]
    fn respond(&self, x: f32) -> f32 {
        let a = self.eval(x);
        let b = self.eval(x * 0.5);
        (a - b) * self.scale + a * b * 0.0625 + self.shift
    }
}

// A generic argument needs no bound to be passed: whether it can is
// decided while tracing.
#[cube]
fn sum_responses<R: Response>(response: &R, x: f32) -> f32 {
    response.respond(x) + response.respond(x + 1.0) + response.respond(x * 2.0)
}

#[cube(launch)]
pub fn kernel_device_fn_methods(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        let a = Curve::new(x, 0.5, 3u32);
        let b = Curve::new(0.25, x, 3u32);
        let c = Curve::new(x * 0.5, x, 5u32);
        output[i] = a.eval(x)
            + b.eval(x)
            + c.eval(x)
            + sum_responses::<Curve>(&a, x)
            + sum_responses::<Curve>(&c, x);
    }
}

#[derive(Clone, Copy)]
struct CurveRef {
    scale: f32,
    shift: f32,
    steps: u32,
}

impl CurveRef {
    fn eval(&self, x: f32) -> f32 {
        let mut y = x * self.scale + self.shift;
        for _ in 0..self.steps {
            y = y * 0.5 + x * self.scale * 0.25 - self.shift * 0.125;
        }
        y
    }

    fn respond(&self, x: f32) -> f32 {
        let a = self.eval(x);
        let b = self.eval(x * 0.5);
        (a - b) * self.scale + a * b * 0.0625 + self.shift
    }

    fn sum_responses(&self, x: f32) -> f32 {
        self.respond(x) + self.respond(x + 1.0) + self.respond(x * 2.0)
    }
}

pub fn test_device_fn_methods<R: Runtime>(client: Client) {
    let n = 64usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02 - 0.5).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(n * core::mem::size_of::<f32>());

    kernel_device_fn_methods::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(n as u32),
        unsafe { BufferArg::from_raw_parts(input_handle, n) },
        unsafe { BufferArg::from_raw_parts(output_handle.clone(), n) },
    );

    let output = client.read_one_unchecked(output_handle);
    let output = f32::from_bytes(&output);
    for (i, &x) in input.iter().enumerate() {
        let curve = |scale, shift, steps| CurveRef {
            scale,
            shift,
            steps,
        };
        let (a, b, c) = (curve(x, 0.5, 3), curve(0.25, x, 3), curve(x * 0.5, x, 5));
        let expected = a.eval(x) + b.eval(x) + c.eval(x) + a.sum_responses(x) + c.sum_responses(x);
        let tolerance = 1e-3 * expected.abs().max(1.0);
        assert!(
            (output[i] - expected).abs() <= tolerance,
            "output[{i}] = {}, expected {expected}",
            output[i]
        );
    }
}

#[cube(inline(never))]
fn weighted_sum(values: &[f32], weights: &[f32], start: usize, count: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in start..start + count {
        acc += values[i] * weights[i % weights.len()] + values[i] * 0.125;
    }
    acc
}

#[cube(inline(never))]
fn scale_into(values: &mut [f32], at: usize, factor: f32, shift: f32) {
    values[at] = values[at] * factor + shift * 0.5 - factor * 0.25 + shift * factor;
}

#[cube(launch)]
pub fn kernel_device_fn_slices(values: &[f32], weights: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        let a = weighted_sum(values, weights, i, 4);
        let b = weighted_sum(values, weights, i + 4, 4);
        output[i] = a;
        scale_into(output, i, b, a);
        scale_into(output, i, a * 0.5, b * 0.25);
    }
}

pub fn test_device_fn_slices<R: Runtime>(client: Client) {
    let n = 32usize;
    let values: Vec<f32> = (0..n + 8).map(|i| (i as f32) * 0.03 - 0.4).collect();
    let weights: Vec<f32> = (0..5).map(|i| 1.0 + i as f32 * 0.5).collect();
    let values_handle = client.create_from_slice(f32::as_bytes(&values));
    let weights_handle = client.create_from_slice(f32::as_bytes(&weights));
    let output_handle = client.empty(n * core::mem::size_of::<f32>());

    kernel_device_fn_slices::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(n as u32),
        unsafe { BufferArg::from_raw_parts(values_handle, n + 8) },
        unsafe { BufferArg::from_raw_parts(weights_handle, 5) },
        unsafe { BufferArg::from_raw_parts(output_handle.clone(), n) },
    );

    let output = client.read_one_unchecked(output_handle);
    let output = f32::from_bytes(&output);
    let weighted = |start: usize| {
        (start..start + 4)
            .map(|i| values[i] * weights[i % weights.len()] + values[i] * 0.125)
            .sum::<f32>()
    };
    let scale =
        |v: f32, factor: f32, shift: f32| v * factor + shift * 0.5 - factor * 0.25 + shift * factor;
    for i in 0..n {
        let (a, b) = (weighted(i), weighted(i + 4));
        let expected = scale(scale(a, b, a), a * 0.5, b * 0.25);
        let tolerance = 1e-3 * expected.abs().max(1.0);
        assert!(
            (output[i] - expected).abs() <= tolerance,
            "output[{i}] = {}, expected {expected}",
            output[i]
        );
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

pub fn test_device_fn_calls<R: Runtime>(client: Client) {
    let n = 96usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let input_handle = client.create_from_slice(f32::as_bytes(&input));
    let output_handle = client.empty(n * core::mem::size_of::<f32>());
    let steps_handle = client.empty(n * core::mem::size_of::<u32>());

    kernel_device_fn_calls::launch(
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

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Pair {
    re: f32,
    im: f32,
}

#[cube(inline(never))]
fn rotate(p: Pair, angle: f32) -> Pair {
    let (s, c) = (f32::sin(angle), f32::cos(angle));
    Pair {
        re: p.re * c - p.im * s,
        im: p.re * s + p.im * c,
    }
}

#[cube(launch)]
pub fn kernel_device_fn_struct_returns(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        let mut p = Pair { re: x, im: x * 0.5 };
        for k in 0..3u32 {
            p = rotate(p, x * 0.1 + f32::cast_from(k));
        }
        output[2 * i] = p.re;
        output[2 * i + 1] = p.im;
    }
}

/// A function returning a struct, called in a loop, matches plain Rust.
pub fn test_device_fn_struct_returns<R: Runtime>(client: Client) {
    let input: Vec<f32> = (0..32).map(|i| i as f32 * 0.25 - 3.0).collect();
    let input_buffer = crate::compute::Buffer::create(&client, &input);
    let output = crate::compute::Buffer::<f32>::empty(&client, 64);
    kernel_device_fn_struct_returns::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        (&input_buffer).into(),
        (&output).into(),
    );
    let output = output.read(&client).unwrap();
    for (i, x) in input.iter().enumerate() {
        let (mut re, mut im) = (*x, x * 0.5);
        for k in 0..3 {
            let angle = x * 0.1 + k as f32;
            let (s, c) = (angle.sin(), angle.cos());
            (re, im) = (re * c - im * s, re * s + im * c);
        }
        for (actual, expected) in [(output[2 * i], re), (output[2 * i + 1], im)] {
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "element {i}: {actual} vs {expected}"
            );
        }
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_device_fn {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_device_fn_struct_returns() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::device_fn::test_device_fn_struct_returns::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_device_fn_calls() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::device_fn::test_device_fn_calls::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_device_fn_methods() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::device_fn::test_device_fn_methods::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_device_fn_slices() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::device_fn::test_device_fn_slices::<TestRuntime>(client);
        }
    };
}
