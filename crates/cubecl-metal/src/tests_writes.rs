//! Host writes take their place in the stream: dispatches encoded before a
//! write read the old bytes and dispatches after it the new ones, without
//! the host waiting for the device in between.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

#[cube(launch)]
fn copy_slowly(input: &[u32], output: &mut [u32], #[comptime] spins: u32) {
    if ABSOLUTE_POS < output.len() {
        // Keeps the GPU busy, so the write below is encoded while this
        // copy has yet to read its input.
        let mut acc = ABSOLUTE_POS as u32;
        for _ in 0..spins {
            acc = acc * 1664525u32 + 1013904223u32;
        }
        output[ABSOLUTE_POS] = select(acc == 0xFFFF_FFFFu32, 0u32, input[ABSOLUTE_POS]);
    }
}

fn copy(client: &Client, input: &Buffer<u32>, output: &Buffer<u32>, n: usize) {
    copy_slowly::launch(
        client,
        CubeCount::Static((n as u32).div_ceil(256), 1, 1),
        CubeDim::new_1d(256),
        input.into(),
        output.into(),
        20_000,
    );
}

#[test]
fn a_write_lands_between_the_dispatches_around_it() {
    let client = R::client(&Default::default());
    let n = 1 << 16;
    let input = Buffer::<u32>::empty(&client, n);
    let before = Buffer::<u32>::empty(&client, n);
    let after = Buffer::<u32>::empty(&client, n);
    input.write(&client, &vec![3; n]);
    copy(&client, &input, &before, n);
    input.write(&client, &vec![5; n]);
    copy(&client, &input, &after, n);
    assert!(before.read(&client).unwrap().iter().all(|&x| x == 3));
    assert!(after.read(&client).unwrap().iter().all(|&x| x == 5));
}

#[test]
fn many_small_writes_keep_their_order() {
    let client = R::client(&Default::default());
    let n = 256;
    let input = Buffer::<u32>::empty(&client, n);
    let outputs: Vec<_> = (0..64).map(|_| Buffer::<u32>::empty(&client, n)).collect();
    for (k, output) in outputs.iter().enumerate() {
        input.write(&client, &vec![k as u32; n]);
        copy(&client, &input, output, n);
    }
    for (k, output) in outputs.iter().enumerate() {
        assert!(
            output.read(&client).unwrap().iter().all(|&x| x == k as u32),
            "copy {k}"
        );
    }
}
