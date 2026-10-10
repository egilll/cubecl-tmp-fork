//! Freed buffers leave the device: a buffer a kernel wrote, once every
//! handle is dropped, is returned by an explicit cleanup rather than held
//! by the record of that write or by the command buffer that made it.

use cubecl_core::MemoryScope;
use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

#[cube(launch)]
fn fill(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 7;
    }
}

#[test]
fn a_written_dedicated_buffer_is_released_by_cleanup() {
    let client = R::client(&Default::default());
    let len = 1 << 20;
    let output = client.memory_dedicated_allocation((), |()| Buffer::<u32>::empty(&client, len));
    fill::launch(
        &client,
        CubeCount::Static(len as u32 / 256, 1, 1),
        CubeDim::new_1d(256),
        (&output).into(),
    );
    assert_eq!(output.slice(0..1).read(&client).unwrap(), [7]);
    let in_use = || {
        client
            .memory_report(MemoryScope::Device)
            .streams
            .iter()
            .map(|stream| stream.pools.dedicated.usage.bytes_in_use)
            .sum::<u64>()
    };
    assert_eq!(in_use(), 4 * len as u64);
    drop(output);
    client.memory_cleanup().unwrap();
    assert_eq!(in_use(), 0);
}
