//! Execution faults: what a failed command buffer reports, and how a stream
//! recovers from one. A real GPU fault can't be caused on demand, so these
//! tests make a command buffer report one: launching a kernel whose name
//! contains `inject_execution_fault` marks its command buffer (tests only).

use cubecl_core::server::{ExecutionFaultKind, ServerError};
use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;

use crate::compute::stream::StreamFault;

type R = crate::MetalRuntime;

#[cube(launch)]
fn add_one(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1;
    }
}

#[cube(launch)]
fn inject_execution_fault(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 2;
    }
}

fn is_fault(err: &ServerError) -> bool {
    format!("{err}").contains("An execution fault (unknown)")
}

#[test]
fn a_stream_recovers_from_an_execution_fault() {
    let client = R::client(&Default::default());
    let input = Buffer::create(&client, &[1u32, 2, 3, 4]);
    let kept = Buffer::<u32>::empty(&client, 4);
    let lost = Buffer::<u32>::empty(&client, 4);

    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&input).into(),
        (&kept).into(),
    );
    assert_eq!(kept.read(&client).unwrap(), vec![2, 3, 4, 5]);

    inject_execution_fault::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&input).into(),
        (&lost).into(),
    );
    let err = lost.read(&client).unwrap_err();
    assert!(is_fault(&err), "{err}");
    // Sticky until reset: every wait on the stream fails.
    assert!(kept.read(&client).is_err());

    client.reset_stream().unwrap();

    // Work that completed before the fault kept its result; what the faulted
    // work was writing reports the fault.
    assert_eq!(kept.read(&client).unwrap(), vec![2, 3, 4, 5]);
    let err = lost.read(&client).unwrap_err();
    assert!(is_fault(&err), "{err}");

    // New work runs, and relaunching the lost write repairs it.
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&input).into(),
        (&lost).into(),
    );
    assert_eq!(lost.read(&client).unwrap(), vec![2, 3, 4, 5]);

    // Resetting a stream with no fault changes nothing.
    client.reset_stream().unwrap();
    assert_eq!(kept.read(&client).unwrap(), vec![2, 3, 4, 5]);
}

#[test]
fn command_buffer_errors_are_classified() {
    let kind = |code, description: &str| StreamFault::classify(code, description, "").kind;
    assert_eq!(
        kind(
            1,
            "Execution of the command buffer was aborted due to an error during execution. \
             Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)"
        ),
        ExecutionFaultKind::Interactivity
    );
    assert_eq!(
        kind(
            1,
            "Innocent Victim (kIOGPUCommandBufferCallbackErrorInnocentVictim)"
        ),
        ExecutionFaultKind::InnocentVictim
    );
    assert_eq!(kind(3, "Page fault"), ExecutionFaultKind::PageFault);
    assert_eq!(
        kind(1, "kIOGPUCommandBufferCallbackErrorPageFault"),
        ExecutionFaultKind::PageFault
    );
    assert_eq!(kind(2, "Timed out"), ExecutionFaultKind::Timeout);
    assert_eq!(
        kind(1, "(kIOGPUCommandBufferCallbackErrorHang)"),
        ExecutionFaultKind::Timeout
    );
    assert_eq!(kind(8, "Out of memory"), ExecutionFaultKind::OutOfMemory);
    assert_eq!(kind(1, "Internal error"), ExecutionFaultKind::Unknown);
}
