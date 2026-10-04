use crate::prelude::*;
use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

// Pin that a kernel without `fast_math` divides and takes square roots correctly rounded, as the
// CPU does. Not part of `testgen_all`: only a backend that promises IEEE semantics by default
// invokes it. On Metal it tells safe math from fast math, which the compile options would
// otherwise decide.

#[cube(launch)]
fn kernel_div_sqrt(lhs: &[f32], rhs: &[f32], quotient: &mut [f32], root: &mut [f32]) {
    if ABSOLUTE_POS < quotient.len() {
        quotient[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] / rhs[ABSOLUTE_POS];
        root[ABSOLUTE_POS] = f32::abs(lhs[ABSOLUTE_POS]).sqrt();
    }
}

pub fn test_div_sqrt_correctly_rounded<R: Runtime>(client: Client) {
    let n = 4096;
    let lhs: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7311).sin() * 100.0 + (i % 7) as f32 * 1e-3)
        .collect();
    let rhs: Vec<f32> = (0..n).map(|i| lhs[i ^ 1] + 0.25).collect();

    let lhs_handle = client.create_from_slice(f32::as_bytes(&lhs));
    let rhs_handle = client.create_from_slice(f32::as_bytes(&rhs));
    let quotient = client.empty(n * size_of::<f32>());
    let root = client.empty(n * size_of::<f32>());
    kernel_div_sqrt::launch(
        &client,
        CubeCount::Static((n as u32).div_ceil(256), 1, 1),
        CubeDim::new_1d(256),
        unsafe { BufferArg::from_raw_parts(lhs_handle, n) },
        unsafe { BufferArg::from_raw_parts(rhs_handle, n) },
        unsafe { BufferArg::from_raw_parts(quotient.clone(), n) },
        unsafe { BufferArg::from_raw_parts(root.clone(), n) },
    );
    let quotient = client.read_one_unchecked(quotient);
    let root = client.read_one_unchecked(root);
    let (quotient, root) = (f32::from_bytes(&quotient), f32::from_bytes(&root));

    let wrong_quotients = (0..n)
        .filter(|&i| quotient[i].to_bits() != (lhs[i] / rhs[i]).to_bits())
        .count();
    let wrong_roots = (0..n)
        .filter(|&i| root[i].to_bits() != lhs[i].abs().sqrt().to_bits())
        .count();
    assert_eq!(
        (wrong_quotients, wrong_roots),
        (0, 0),
        "quotients and roots of {n} that differ from the correctly rounded value"
    );
}

#[cube(launch)]
fn kernel_two_sum(lhs: &[f32], rhs: &[f32], sum: &mut [f32], error: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < sum.len() {
        let (a, b) = (lhs[i], rhs[i]);
        let s = a + b;
        let bb = s - a;
        sum[i] = s;
        error[i] = (a - (s - bb)) + (b - bb);
    }
}

/// TwoSum's error term is exact only if every operation is rounded on its
/// own: no reassociation, no contraction. Pins that the IEEE default holds
/// for arithmetic, not just for single operations.
pub fn test_two_sum_is_exact<R: Runtime>(client: Client) {
    let n = 4096;
    let lhs: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.7311).sin() * 1e4 + (i % 7) as f32 * 1e-3)
        .collect();
    let rhs: Vec<f32> = (0..n).map(|i| (i as f32 * 1.37).cos() * 1e-4).collect();
    let lhs_buffer = crate::compute::Buffer::create(&client, &lhs);
    let rhs_buffer = crate::compute::Buffer::create(&client, &rhs);
    let sum = crate::compute::Buffer::<f32>::empty(&client, n);
    let error = crate::compute::Buffer::<f32>::empty(&client, n);
    kernel_two_sum::launch(
        &client,
        CubeCount::Static((n as u32).div_ceil(256), 1, 1),
        CubeDim::new_1d(256),
        (&lhs_buffer).into(),
        (&rhs_buffer).into(),
        (&sum).into(),
        (&error).into(),
    );
    let (sum, error) = (sum.read(&client).unwrap(), error.read(&client).unwrap());
    let wrong = (0..n)
        .filter(|&i| {
            let (a, b) = (lhs[i], rhs[i]);
            let s = a + b;
            let bb = s - a;
            let e = (a - (s - bb)) + (b - bb);
            sum[i].to_bits() != s.to_bits() || error[i].to_bits() != e.to_bits()
        })
        .count();
    assert_eq!(wrong, 0, "TwoSum results of {n} that differ from the CPU's");
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_ieee_rounding {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_div_sqrt_correctly_rounded() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::ieee_rounding::test_div_sqrt_correctly_rounded::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_two_sum_is_exact() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::ieee_rounding::test_two_sum_is_exact::<TestRuntime>(
                client,
            );
        }
    };
}
