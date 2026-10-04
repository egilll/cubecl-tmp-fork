use crate::compute::Buffer;
use crate::prelude::*;
use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

#[cube(launch)]
fn kernel_scale(input: &[f32], output: &mut [f32], factor: f32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * factor;
    }
}

/// A launch through typed buffers, including a slice of one, writes what a
/// raw-parts launch over the same memory writes.
pub fn test_typed_buffer_launch<R: Runtime>(client: Client) {
    let data: Vec<f32> = (0..64).map(|i| i as f32 - 20.0).collect();
    let input = Buffer::create(&client, &data);
    let output = Buffer::<f32>::empty(&client, 64);
    kernel_scale::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        (&input).into(),
        (&output).into(),
        3.0,
    );

    let raw = client.empty(64 * size_of::<f32>());
    kernel_scale::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        unsafe { BufferArg::from_raw_parts(input.handle().clone(), 64) },
        unsafe { BufferArg::from_raw_parts(raw.clone(), 64) },
        3.0,
    );
    let raw = f32::from_bytes(&client.read_one(raw).unwrap()).to_vec();
    assert_eq!(output.read(&client).unwrap(), raw);

    // A slice reads and writes only its own elements.
    let window = output.slice(16..48);
    assert_eq!(window.len(), 32);
    kernel_scale::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(64),
        (&input.slice(0..32)).into(),
        (&window).into(),
        0.5,
    );
    let all = output.read(&client).unwrap();
    for (i, value) in all.iter().enumerate() {
        let expected = match i {
            16..48 => data[i - 16] * 0.5,
            _ => data[i] * 3.0,
        };
        assert_eq!(*value, expected, "element {i}");
    }
    assert_eq!(window.read(&client).unwrap(), all[16..48].to_vec());
}

/// A write through a slice replaces exactly that slice's leading elements.
pub fn test_typed_buffer_write<R: Runtime>(client: Client) {
    let buffer = Buffer::create(&client, &[0u32; 8]);
    buffer.slice(3..6).write(&client, &[7, 8, 9]);
    buffer.slice(6..8).write(&client, &[1]);
    buffer.write(&client, &[]);
    assert_eq!(buffer.read(&client).unwrap(), [0, 0, 0, 7, 8, 9, 1, 0]);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_typed_buffer {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_typed_buffer_launch() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::typed_buffer::test_typed_buffer_launch::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_typed_buffer_write() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::typed_buffer::test_typed_buffer_write::<TestRuntime>(
                client,
            );
        }
    };
}
