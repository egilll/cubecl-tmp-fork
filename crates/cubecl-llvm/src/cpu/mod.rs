//! CPU target.

pub mod abi;
pub mod entrypoint;
pub mod f16_evaluation;
pub mod jit;
#[cfg(test)]
mod offline_tests;
pub mod plane;
pub mod uniform_load;
pub mod unit_loops;
