//! Device-wide primitives for `CubeCL`: prefix sums, stream compaction, an
//! indirect-dispatch helper, radix sort, segmented reduction and FFT.
//!
//! Every algorithm here uses only shared memory and cube barriers, so it runs
//! on every backend (Metal, Vulkan, WebGPU, CUDA), and none relies on
//! cubes waiting on each other, which GPUs without a forward-progress
//! guarantee (Apple, WebGPU) can't promise.

pub mod fft;
pub mod reduce;
pub mod scan;
pub mod sort;

#[cfg(feature = "export_tests")]
pub mod tests;
