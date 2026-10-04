//! Launch-validation tests for the Metal backend.
//!
//! Resource-limit violations must surface as `LaunchError::TooManyResources`, not as the
//! opaque `CompilationError` Metal's pipeline creation produces when a kernel genuinely
//! uses more threadgroup memory than the device allows.

use cubecl_core::server::{LaunchError, ResourceLimitError, ServerError};
use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

#[cube(launch)]
fn oversized_smem_kernel(output: &mut [u32], #[comptime] shared_size: usize) {
    let mut shared = Shared::new_slice(shared_size);
    // Runtime-dependent indices so the MSL compiler can't shrink the allocation.
    let idx = output[0] as usize;
    shared[idx] = output[0];
    sync_cube();
    output[0] = shared[idx + 1];
}

#[test]
fn oversized_shared_memory_is_a_resource_limit_error() {
    let client = R::client(&Default::default());
    let max = client.properties().hardware.max_shared_memory_size;
    let shared_size = (max + 1).div_ceil(size_of::<u32>());
    let requested_bytes = shared_size * size_of::<u32>();

    let handle = client.create_from_slice(u32::as_bytes(&[0]));
    oversized_smem_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        shared_size,
    );

    // A launch failure is not the flush's to report: it lives on the buffer
    // the launch never wrote, and the read of it is what surfaces it.
    client
        .flush()
        .expect("a launch failure is not the flush's to report");

    let err = client
        .read_one(handle)
        .expect_err("the launch never wrote the buffer, so the read fails on it");
    let ServerError::Several { mut errors, .. } = err else {
        panic!("a read of an unwritten buffer reports its failures, got: {err}");
    };
    let root = match errors.remove(0) {
        ServerError::Unwritten { root, .. } => root,
        other => panic!("expected the read to fail on the unwritten buffer, got: {other}"),
    };
    match *root {
        ServerError::Launch(LaunchError::TooManyResources(ResourceLimitError::SharedMemory {
            requested,
            max: reported_max,
            ..
        })) => {
            assert_eq!(requested, requested_bytes);
            assert_eq!(reported_max, max);
        }
        other => panic!("expected a shared memory resource limit error, got: {other}"),
    }
}

/// Reads the first element of every input into the output, so every input is a
/// binding of its own.
#[cube(launch_unchecked)]
fn gather_first(inputs: &Sequence<Box<[u32]>>, output: &mut [u32]) {
    let mut total = 0u32;
    #[unroll]
    for i in 0..inputs.len() {
        total += inputs[i][0];
    }
    output[0] = total;
}

fn launch_gather_first(client: &Client, inputs: usize) -> Result<u32, ServerError> {
    let inputs: Vec<_> = (0..inputs)
        .map(|_| client.create_from_slice(u32::as_bytes(&[1])))
        .collect();
    let output = client.create_from_slice(u32::as_bytes(&[0]));
    unsafe {
        gather_first::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            inputs
                .iter()
                .map(|input| BufferArg::from_raw_parts(input.clone(), 1))
                .collect(),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }
    client
        .read_one(output)
        .map(|bytes| u32::from_bytes(&bytes[..])[0])
}

/// With the output, `max_bindings` buffers runs and one more is refused.
#[test]
fn max_bindings_is_the_limit() {
    let client = R::client(&Default::default());
    let max = client.properties().hardware.max_bindings as usize;

    let total = launch_gather_first(&client, max - 1).expect("the kernel is within the limit");
    assert_eq!(total, max as u32 - 1);

    let err = launch_gather_first(&client, max).expect_err("the kernel is over the limit");
    assert!(err.is_refusal(), "expected a refusal, got: {err}");
}

#[cube(launch, create_dummy_kernel)]
fn limited_kernel(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = f32::cast_from(ABSOLUTE_POS);
    }
}

#[test]
fn a_kernels_pipeline_reports_its_own_limits() {
    let client = R::client(&Default::default());
    let kernel = limited_kernel::create_dummy_kernel(
        client.properties_shared(),
        client.target_properties_shared(),
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        (&Buffer::<f32>::empty(&client, 64)).into(),
    );
    let limits = client.pipeline_limits(Box::new(kernel)).unwrap();
    let hardware = &client.properties().hardware;
    assert!(limits.max_units_per_cube >= 64);
    assert!(limits.max_units_per_cube <= hardware.max_units_per_cube);
    // Apple GPUs run 32-wide SIMD groups.
    assert_eq!(limits.plane_size, 32);
}
