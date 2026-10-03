use crate::prelude::*;
use crate::{self as cubecl};
use cubecl_runtime::runtime::Runtime;
use cubecl_runtime::server::Handle;

// Pin `||` / `&&` short-circuit semantics. The RHS mutates a side-channel.
// If short-circuit holds, the channel is never written.

#[cube]
fn mark_side_channel(side_channel: &mut Array<u32>) -> bool {
    side_channel[0] = 1u32;
    false
}

#[cube(launch)]
pub fn kernel_short_circuit_or(output: &mut [u32], left_input: u32) {
    if UNIT_POS == 0 {
        let mut side_channel = Array::<u32>::new(1usize);
        side_channel[0] = 0u32;
        let flag = (left_input != 0u32) || mark_side_channel(&mut side_channel);
        if flag {
            output[0] = side_channel[0];
        } else {
            output[0] = 999u32;
        }
    }
}

#[cube(launch)]
pub fn kernel_short_circuit_and(output: &mut [u32], left_input: u32) {
    if UNIT_POS == 0 {
        let mut side_channel = Array::<u32>::new(1usize);
        side_channel[0] = 0u32;
        let flag = (left_input != 0u32) && mark_side_channel(&mut side_channel);
        if !flag {
            output[0] = side_channel[0];
        } else {
            output[0] = 999u32;
        }
    }
}

// Pure operands take the eager path. The logical result must still be correct.

#[cube(launch)]
pub fn kernel_pure_or(output: &mut [u32], a: u32, b: u32) {
    if UNIT_POS == 0 {
        let flag = (a != 0u32) || (b != 0u32);
        if flag {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

#[cube(launch)]
pub fn kernel_pure_and(output: &mut [u32], a: u32, b: u32) {
    if UNIT_POS == 0 {
        let flag = (a != 0u32) && (b != 0u32);
        if flag {
            output[0] = 1u32;
        } else {
            output[0] = 0u32;
        }
    }
}

// A short-circuit operand that reads memory, used as a loop condition. The right-hand side is
// lowered into a branch inside the condition, which must stay inside the loop.

#[cube(launch)]
pub fn kernel_while_and_read(data: &[u32], output: &mut [u32], end: u32) {
    let mut i = 0u32;
    while i < end && data[i as usize] == 0u32 {
        i += 1;
    }
    output[0] = i;
}

#[cube(launch)]
pub fn kernel_while_or_read(data: &[u32], output: &mut [u32], skip: u32) {
    let mut i = 0u32;
    while i < skip || data[i as usize] == 0u32 {
        i += 1;
    }
    output[0] = i;
}

pub fn test_short_circuit_or<R: Runtime>(client: Client) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_or::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        1u32,
    );
    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual[0], 0, "`||` did not short-circuit");
}

pub fn test_short_circuit_and<R: Runtime>(client: Client) {
    let handle = client.empty(core::mem::size_of::<u32>());
    kernel_short_circuit_and::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
        0u32,
    );
    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);
    assert_eq!(actual[0], 0, "`&&` did not short-circuit");
}

fn run_pure(client: &Client, launch: impl Fn(&Client, Handle, u32, u32), a: u32, b: u32) -> u32 {
    let handle = client.empty(core::mem::size_of::<u32>());
    launch(client, handle.clone(), a, b);
    let actual = client.read_one_unchecked(handle);
    u32::from_bytes(&actual)[0]
}

pub fn test_pure_or<R: Runtime>(client: Client) {
    let launch = |client: &Client, handle: Handle, a, b| {
        kernel_pure_or::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handle, 1) },
            a,
            b,
        );
    };
    assert_eq!(run_pure(&client, launch, 0, 0), 0, "0 || 0");
    assert_eq!(run_pure(&client, launch, 0, 7), 1, "0 || 7");
    assert_eq!(run_pure(&client, launch, 5, 0), 1, "5 || 0");
    assert_eq!(run_pure(&client, launch, 5, 7), 1, "5 || 7");
}

pub fn test_pure_and<R: Runtime>(client: Client) {
    let launch = |client: &Client, handle: Handle, a, b| {
        kernel_pure_and::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(handle, 1) },
            a,
            b,
        );
    };
    assert_eq!(run_pure(&client, launch, 0, 0), 0, "0 && 0");
    assert_eq!(run_pure(&client, launch, 0, 7), 0, "0 && 7");
    assert_eq!(run_pure(&client, launch, 5, 0), 0, "5 && 0");
    assert_eq!(run_pure(&client, launch, 5, 7), 1, "5 && 7");
}

fn run_while_read(
    client: &Client,
    launch: impl Fn(&Client, Handle, Handle, u32),
    data: &[u32],
    arg: u32,
) -> u32 {
    let data = client.create_from_slice(u32::as_bytes(data));
    let output = client.create_from_slice(u32::as_bytes(&[u32::MAX]));
    launch(client, data, output.clone(), arg);
    let actual = client.read_one_unchecked(output);
    u32::from_bytes(&actual)[0]
}

pub fn test_while_and_read<R: Runtime>(client: Client) {
    let launch = |client: &Client, data: Handle, output: Handle, end| {
        kernel_while_and_read::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(data, 6) },
            unsafe { BufferArg::from_raw_parts(output, 1) },
            end,
        );
    };
    let data = [0, 0, 0, 1, 0, 0];
    assert_eq!(
        run_while_read(&client, launch, &data, 6),
        3,
        "stops at data"
    );
    assert_eq!(run_while_read(&client, launch, &data, 2), 2, "stops at end");
}

pub fn test_while_or_read<R: Runtime>(client: Client) {
    let launch = |client: &Client, data: Handle, output: Handle, skip| {
        kernel_while_or_read::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(data, 6) },
            unsafe { BufferArg::from_raw_parts(output, 1) },
            skip,
        );
    };
    let data = [1, 0, 0, 1, 0, 1];
    assert_eq!(
        run_while_read(&client, launch, &data, 0),
        0,
        "stops at data"
    );
    assert_eq!(
        run_while_read(&client, launch, &data, 1),
        3,
        "skips, then stops at data"
    );
    assert_eq!(
        run_while_read(&client, launch, &data, 4),
        5,
        "skips past data"
    );
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_short_circuit {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_short_circuit_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_short_circuit_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_short_circuit_and::<TestRuntime>(
                client,
            );
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_while_and_read() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_while_and_read::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_while_or_read() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_while_or_read::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_or() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_pure_or::<TestRuntime>(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_pure_and() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::short_circuit::test_pure_and::<TestRuntime>(client);
        }
    };
}
