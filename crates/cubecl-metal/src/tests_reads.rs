//! Reads wait only for the work that writes what they read, and a read still
//! pending returns the bytes as of when it was requested, whatever is
//! launched after it.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

type R = crate::MetalRuntime;

#[cube(launch)]
fn fill(output: &mut [u32], value: u32, #[comptime] spins: u32) {
    if ABSOLUTE_POS < output.len() {
        // Keeps the GPU busy, so the read below is requested before the
        // write it waits for has run. The result can't be folded away, and
        // the sentinel is never reached in practice.
        let mut acc = ABSOLUTE_POS as u32;
        for _ in 0..spins {
            acc = acc * 1664525u32 + 1013904223u32;
        }
        output[ABSOLUTE_POS] = select(acc == 0xFFFF_FFFFu32, 0u32, value);
    }
}

#[test]
fn a_pending_read_returns_the_bytes_as_of_its_request() {
    let client = R::client(&Default::default());
    let n = 1 << 16;
    let buffer = Buffer::<u32>::empty(&client, n);
    let launch = |value: u32| {
        fill::launch(
            &client,
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            (&buffer).into(),
            value,
            20_000,
        )
    };
    launch(7);
    let read = client.read_async(vec![buffer.handle().clone()]);
    // Overwrites the same bytes before the read resolves.
    launch(9);
    let first = cubecl_core::future::block_on(read).unwrap();
    assert!(u32::from_bytes(&first[0]).iter().all(|&x| x == 7));
    assert!(buffer.read(&client).unwrap().iter().all(|&x| x == 9));
}

#[test]
fn a_read_of_completed_work_needs_no_wait() {
    let client = R::client(&Default::default());
    let done = Buffer::<u32>::empty(&client, 256);
    let busy = Buffer::<u32>::empty(&client, 1 << 16);
    fill::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(256),
        (&done).into(),
        3,
        1,
    );
    cubecl_core::future::block_on(client.sync()).unwrap();
    // Unrelated work in flight.
    fill::launch(
        &client,
        CubeCount::Static(256, 1, 1),
        CubeDim::new_1d(256),
        (&busy).into(),
        5,
        20_000,
    );
    let mut read = client.sync_buffers([done.handle()]);
    let waker = std::task::Waker::noop();
    let mut cx = std::task::Context::from_waker(waker);
    assert!(
        read.as_mut().poll(&mut cx).is_ready(),
        "a sync of buffers nothing pending writes resolves at once"
    );
    assert!(done.read(&client).unwrap().iter().all(|&x| x == 3));
    assert!(busy.read(&client).unwrap().iter().all(|&x| x == 5));
}

#[test]
fn in_flight_work_is_counted_until_it_completes() {
    let client = R::client(&Default::default()).lane(20);
    let buffer = Buffer::<u32>::empty(&client, 1 << 16);
    for value in 0..8 {
        fill::launch(
            &client,
            CubeCount::Static(256, 1, 1),
            CubeDim::new_1d(256),
            (&buffer).into(),
            value,
            20_000,
        );
    }
    // The batches may already have run by the time a count is read, so only
    // the drain below is certain; completion handlers run after the sync's
    // event, so wait for the count rather than read it right away.
    cubecl_core::future::block_on(client.sync()).unwrap();
    cubecl_core::future::block_on(client.in_flight_below(1));
    assert_eq!(client.in_flight(), Default::default());
    assert!(buffer.read(&client).unwrap().iter().all(|&x| x == 7));
}
