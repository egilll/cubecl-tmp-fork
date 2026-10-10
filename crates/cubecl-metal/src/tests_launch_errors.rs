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

/// Without graph support the backend refuses at preparation, before a
/// caller's warmup run would execute work it can never record.
#[test]
fn graph_preparation_is_refused_up_front() {
    let client = <crate::MetalRuntime as cubecl_server::runtime::Runtime>::client(&Default::default());
    assert!(client.graph_prepare().is_err());
    assert!(!client.is_capturing());
}


#[cube(launch)]
fn thirty_two_buffers(
    b0: &[f32],
    b1: &[f32],
    b2: &[f32],
    b3: &[f32],
    b4: &[f32],
    b5: &[f32],
    b6: &[f32],
    b7: &[f32],
    b8: &[f32],
    b9: &[f32],
    b10: &[f32],
    b11: &[f32],
    b12: &[f32],
    b13: &[f32],
    b14: &[f32],
    b15: &[f32],
    b16: &[f32],
    b17: &[f32],
    b18: &[f32],
    b19: &[f32],
    b20: &[f32],
    b21: &[f32],
    b22: &[f32],
    b23: &[f32],
    b24: &[f32],
    b25: &[f32],
    b26: &[f32],
    b27: &[f32],
    b28: &[f32],
    b29: &[f32],
    b30: &[f32],
    b31: &[f32],
    output: &mut [f32],
) {
    output[0] = b0[0] + b1[0] + b2[0] + b3[0] + b4[0] + b5[0] + b6[0] + b7[0] + b8[0] + b9[0] + b10[0] + b11[0] + b12[0] + b13[0] + b14[0] + b15[0] + b16[0] + b17[0] + b18[0] + b19[0] + b20[0] + b21[0] + b22[0] + b23[0] + b24[0] + b25[0] + b26[0] + b27[0] + b28[0] + b29[0] + b30[0] + b31[0];
}

/// A kernel binding more buffers than a Metal function takes is refused
/// with the count, not left to fail inside the MSL compiler.
#[test]
fn too_many_buffers_are_a_validation_error() {
    let client = R::client(&Default::default());
    let handles: Vec<_> = (0..32).map(|_| client.create_from_slice(f32::as_bytes(&[1.0]))).collect();
    let output = client.create_from_slice(f32::as_bytes(&[0.0]));
    thirty_two_buffers::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handles[0].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[1].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[2].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[3].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[4].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[5].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[6].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[7].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[8].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[9].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[10].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[11].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[12].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[13].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[14].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[15].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[16].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[17].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[18].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[19].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[20].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[21].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[22].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[23].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[24].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[25].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[26].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[27].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[28].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[29].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[30].clone(), 1) },
            unsafe { BufferArg::from_raw_parts(handles[31].clone(), 1) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
    );
    let error = client.read_one(output).expect_err("the launch never wrote the output").to_string();
    assert!(error.contains("binds") && error.contains("Metal binds at most 31"), "{error}");
}
