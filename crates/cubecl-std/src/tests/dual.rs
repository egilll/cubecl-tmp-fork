use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::runtime::Runtime;

use crate::dual::{Dual, Real};

/// Written once, over any `Real`.
#[cube]
fn formula<R: Real>(x: R, y: R) -> R {
    R::exp(x * y) + R::sin(x) / R::sqrt(y) - R::constant(0.5) * R::ln(y)
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
        }
    };
}
