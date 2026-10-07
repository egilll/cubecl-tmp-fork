//! Cubecl standard library.
extern crate alloc;

mod fast_math;
mod reinterpret_slice;
mod swizzle;

pub use fast_math::*;
pub use reinterpret_slice::*;
pub use swizzle::*;

mod trigonometry;
pub use trigonometry::*;

/// Quantization functionality required in views
/// Cube-wide reductions over typed values, subgroup first.
pub mod collective;
/// Forward-mode dual numbers, and formulas generic over a real type.
pub mod dual;
/// Accurate arithmetic without `f64`: compensated sums and double-f32.
pub mod numeric;
pub mod quant;
pub mod tensor;

/// Event utilities.
pub mod event;

/// Throughput utilities.
pub mod throughput;

#[cfg(feature = "export_tests")]
pub mod tests;
