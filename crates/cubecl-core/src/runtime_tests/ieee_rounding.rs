use crate::prelude::*;
use alloc::vec::Vec;
use crate::{self as cubecl};
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
    };
}
