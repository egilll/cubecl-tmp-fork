// Metal only exists on Apple platforms; elsewhere the crate compiles to
// nothing so workspace-wide builds and clippy work on any host. The Metal
// bindings it needs are target-gated in Cargo.toml to match.
#![cfg(target_vendor = "apple")]

pub mod compute;
pub mod device;
pub mod memory;
pub mod runtime;
#[cfg(feature = "wgpu")]
pub mod wgpu_interop;

pub use device::{MetalDevice, register_device};
pub use runtime::MetalRuntime;

pub(crate) type MetalCompiler = cubecl_cpp::shared::CppCompiler<cubecl_cpp::target::Metal>;

#[cfg(test)]
mod tests_bf16_cast;
#[cfg(test)]
mod tests_constant_loop_index;
#[cfg(test)]
mod tests_expm1;
#[cfg(test)]
mod tests_faults;
#[cfg(test)]
mod tests_hazards;
#[cfg(test)]
mod tests_launch_errors;
#[cfg(test)]
mod tests_lease;
#[cfg(test)]
mod tests_multistream;
#[cfg(test)]
mod tests_reads;
#[cfg(test)]
mod tests_release;
#[cfg(test)]
mod tests_writes;

#[cfg(test)]
mod tests {
    pub type TestRuntime = crate::MetalRuntime;

    use half::{bf16, f16};

    cubecl_core::testgen_all!(f32: [f16, bf16, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);
    cubecl_core::testgen_profiling!();
    // No `testgen_profiling_nested!`: a stream collects the command buffers of
    // one window at a time (`Stream::profiling`), so an inner `start_profile`
    // replaces the outer's collector and the outer window ends up measuring
    // nothing. Add it once the collector is a stack.

    mod ieee_rounding {
        use super::*;
        cubecl_core::testgen_ieee_rounding!();
    }

    // c32 is lowered to a float2; c64 needs f64, which Metal lacks.
    cubecl_core::testgen_complex_core!(cf32);
    cubecl_core::testgen_complex_compare!(cf32);
    cubecl_core::testgen_complex_math!(cf32);
    cubecl_std::testgen_numeric!();
    cubecl_algorithms::testgen!();
}
