use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::runtime::Runtime;

use crate::dual::{Dual, Real};

/// Written once, over any `Real`.
#[cube]
fn formula<R: Real>(x: R, y: R) -> R {
    R::exp(x * y) + R::sin(x) / R::sqrt(y) - R::constant(0.5) * R::ln(y)
}

/// A sum over runtime terms, accumulated in a mutable `R`, with inputs held
/// in a sequence.
#[cube]
fn series<R: Real>(inputs: &Sequence<R>, terms: u32) -> R {
    let x = *inputs.index(0usize);
    let y = *inputs.index(1usize);
    let mut sum = R::constant(0.0);
    for k in 0..terms {
        sum = sum + R::sin(x * R::constant(f32::cast_from(k + 1))) * y;
    }
    sum
}

#[cube(launch)]
fn kernel_series(input: &[f32], output: &mut [f32], terms: u32) {
    if ABSOLUTE_POS == 0 {
        let mut plain = Sequence::<f32>::new();
        plain.push(input[0]);
        plain.push(input[1]);
        let mut dual = Sequence::<Dual<Const<2>>>::new();
        dual.push(Dual::<Const<2>>::variable(input[0], 0usize));
        dual.push(Dual::<Const<2>>::variable(input[1], 1usize));
        let value = series::<f32>(&plain, terms);
        let derivative = series::<Dual<Const<2>>>(&dual, terms);
        output[0] = value;
        output[1] = derivative.value;
        output[2] = derivative.tangent.extract(0usize);
        output[3] = derivative.tangent.extract(1usize);
    }
}

/// A loop-accumulated formula differentiates like a closed one.
pub fn test_dual_accumulation<R: Runtime>(client: Client) {
    let (x, y) = (0.3f32, 1.7f32);
    let input = Buffer::create(&client, &[x, y]);
    let output = Buffer::<f32>::empty(&client, 4);
    kernel_series::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        (&input).into(),
        (&output).into(),
        5,
    );
    let output = output.read(&client).unwrap();
    let (x, y) = (x as f64, y as f64);
    let value: f64 = (1..=5).map(|k| (x * k as f64).sin() * y).sum();
    let dx: f64 = (1..=5).map(|k| k as f64 * (x * k as f64).cos() * y).sum();
    let dy: f64 = (1..=5).map(|k| (x * k as f64).sin()).sum();
    for (actual, expected) in [(output[0], value), (output[2], dx), (output[3], dy)] {
        assert!(
            (actual as f64 - expected).abs() < 1e-4,
            "{actual} vs {expected}"
        );
    }
    assert_eq!(output[0], output[1]);
}

#[cube(launch)]
fn kernel_formula(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < input.len() / 2 {
        let (x, y) = (input[2 * i], input[2 * i + 1]);
        let plain = formula::<f32>(x, y);
        let dual = formula::<Dual<Const<2>>>(
            Dual::<Const<2>>::variable(x, 0usize),
            Dual::<Const<2>>::variable(y, 1usize),
        );
        output[4 * i] = plain;
        output[4 * i + 1] = dual.value;
        output[4 * i + 2] = dual.tangent.extract(0usize);
        output[4 * i + 3] = dual.tangent.extract(1usize);
    }
}

/// The same formula gives its value as f32 and as a dual number, and the
/// dual's tangents are its partial derivatives.
pub fn test_dual_derivatives<R: Runtime>(client: Client) {
    let points: Vec<f32> = (0..32)
        .flat_map(|i| [0.1 + i as f32 * 0.03, 0.5 + i as f32 * 0.07])
        .collect();
    let input = Buffer::create(&client, &points);
    let output = Buffer::<f32>::empty(&client, 128);
    kernel_formula::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(32),
        (&input).into(),
        (&output).into(),
    );
    let output = output.read(&client).unwrap();
    for i in 0..32 {
        let (x, y) = (points[2 * i] as f64, points[2 * i + 1] as f64);
        let value = (x * y).exp() + x.sin() / y.sqrt() - 0.5 * y.ln();
        let dx = y * (x * y).exp() + x.cos() / y.sqrt();
        let dy = x * (x * y).exp() - 0.5 * x.sin() * y.powf(-1.5) - 0.5 / y;
        let close = |actual: f32, expected: f64| {
            (actual as f64 - expected).abs() <= 1e-4 * expected.abs().max(1.0)
        };
        assert!(close(output[4 * i], value), "value {i}");
        assert_eq!(output[4 * i], output[4 * i + 1], "f32 and dual values {i}");
        assert!(
            close(output[4 * i + 2], dx),
            "d/dx {i}: {} vs {dx}",
            output[4 * i + 2]
        );
        assert!(
            close(output[4 * i + 3], dy),
            "d/dy {i}: {} vs {dy}",
            output[4 * i + 3]
        );
    }
}

#[macro_export]
macro_rules! testgen_dual {
    () => {
        mod dual {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_dual_derivatives() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::dual::test_dual_derivatives::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_dual_accumulation() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::dual::test_dual_accumulation::<TestRuntime>(client);
            }
        }
    };
}
