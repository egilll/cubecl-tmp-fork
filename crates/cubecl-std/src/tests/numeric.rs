use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::runtime::Runtime;

use crate::numeric::{Compensated, DoubleF32};

#[cube(launch)]
fn kernel_sums(input: &[f32], output: &mut [f32]) {
    if UNIT_POS == 0 {
        let mut naive = 0.0f32;
        let mut compensated = Compensated::new();
        let mut double = DoubleF32::from_f32(0.0);
        for i in 0..input.len() {
            naive += input[i];
            compensated.add(input[i]);
            double = double + DoubleF32::from_f32(input[i]);
        }
        output[0] = naive;
        output[1] = compensated.value();
        output[2] = double.hi;
        output[3] = double.lo;
    }
}

#[cube(launch)]
fn kernel_products(lhs: &[f32], rhs: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < lhs.len() {
        let product = DoubleF32::from_f32(lhs[i]) * DoubleF32::from_f32(rhs[i]);
        output[2 * i] = product.hi;
        output[2 * i + 1] = product.lo;
    }
}

fn sums(client: &Client, input: &[f32]) -> [f32; 4] {
    let input_buffer = Buffer::create(client, input);
    let output = Buffer::<f32>::empty(client, 4);
    kernel_sums::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        (&input_buffer).into(),
        (&output).into(),
    );
    output.read(client).unwrap().try_into().unwrap()
}

/// Cancelling terms around a large value: a plain f32 sum loses the small
/// ones, a compensated one keeps them.
pub fn test_compensated_sum<R: Runtime>(client: Client) {
    let mut input = Vec::new();
    for i in 0..1000 {
        input.extend([1.0e8f32, 1.0 + i as f32 * 1e-3, -1.0e8]);
    }
    let exact: f64 = input.iter().map(|&x| x as f64).sum();
    let [naive, compensated, ..] = sums(&client, &input);
    assert!(
        (naive as f64 - exact).abs() > 100.0,
        "the plain sum should drift: {naive} vs {exact}"
    );
    assert!(
        (compensated as f64 - exact).abs() <= exact.abs() * 1e-6,
        "compensated {compensated} vs {exact}"
    );
}

/// A long accumulation in double-f32 stays close to the f64 result.
pub fn test_double_f32_sum<R: Runtime>(client: Client) {
    let input: Vec<f32> = (0..10_000).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let exact: f64 = input.iter().map(|&x| x as f64).sum();
    let [naive, _, hi, lo] = sums(&client, &input);
    let double = DoubleF32 { hi, lo }.to_f64();
    let naive_error = (naive as f64 - exact).abs();
    let double_error = (double - exact).abs();
    assert!(
        double_error * 1000.0 < naive_error.max(1e-9),
        "double-f32 error {double_error:e}, f32 error {naive_error:e}"
    );
}

/// The product of two f32s fits in a double-f32 exactly.
pub fn test_double_f32_product_is_exact<R: Runtime>(client: Client) {
    let lhs: Vec<f32> = (0..256).map(|i| 1.0 + i as f32 * 0.123_457).collect();
    let rhs: Vec<f32> = (0..256).map(|i| 3.0 - i as f32 * 0.017_771).collect();
    let lhs_buffer = Buffer::create(&client, &lhs);
    let rhs_buffer = Buffer::create(&client, &rhs);
    let output = Buffer::<f32>::empty(&client, 512);
    kernel_products::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(256),
        (&lhs_buffer).into(),
        (&rhs_buffer).into(),
        (&output).into(),
    );
    let output = output.read(&client).unwrap();
    for i in 0..256 {
        let exact = lhs[i] as f64 * rhs[i] as f64;
        let got = DoubleF32 {
            hi: output[2 * i],
            lo: output[2 * i + 1],
        }
        .to_f64();
        assert_eq!(got, exact, "product {i}");
    }
}

/// Only for backends that round every float operation on its own (native
/// Metal, wgpu's MSL and SPIR-V routes): WGSL through naga is compiled with
/// Metal's fast math on Apple GPUs, which reassociates the error terms away.
#[macro_export]
macro_rules! testgen_numeric {
    () => {
        mod numeric {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_compensated_sum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::numeric::test_compensated_sum::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_double_f32_sum() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::numeric::test_double_f32_sum::<TestRuntime>(client);
            }

            #[$crate::tests::test_log::test]
            fn test_double_f32_product_is_exact() {
                let client = TestRuntime::client(&Default::default());
                cubecl_std::tests::numeric::test_double_f32_product_is_exact::<TestRuntime>(client);
            }
        }
    };
}
