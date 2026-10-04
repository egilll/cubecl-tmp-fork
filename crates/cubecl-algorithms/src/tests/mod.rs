//! Runtime tests against host references, generated per backend.

pub use test_log;

pub mod fft;
pub mod reduce;
pub mod scan;
pub mod sort;

#[macro_export]
macro_rules! testgen {
    () => {
        mod test_cubecl_algorithms {
            use super::*;

            cubecl_algorithms::testgen_scan!();
            cubecl_algorithms::testgen_sort!();
            cubecl_algorithms::testgen_reduce!();
            cubecl_algorithms::testgen_fft!();
        }
    };
}
