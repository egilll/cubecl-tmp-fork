//! A lease hands a completed buffer to another consumer, pinned and
//! immutable until it is dropped.

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_server::runtime::Runtime;
use objc2_metal::MTLBuffer;

use crate::compute::server::MetalServer;

type R = crate::MetalRuntime;

#[cube(launch)]
fn add_one(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + 1;
    }
}

#[test]
fn a_lease_holds_the_completed_bytes_in_place() {
    let client = R::client(&Default::default());
    let input = Buffer::create(&client, &[1u32, 2, 3, 4]);
    let output = Buffer::<u32>::empty(&client, 4);
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&input).into(),
        (&output).into(),
    );
    let lease =
        cubecl_core::future::block_on(client.export::<MetalServer>(output.handle())).unwrap();

    // The work had completed: the bytes are there, at the leased region.
    let resource = lease.resource();
    assert_eq!(resource.size(), 16);
    let bytes = unsafe {
        let base = resource.inner().contents().as_ptr() as *const u8;
        std::slice::from_raw_parts(base.add(resource.offset() as usize), 16)
    };
    assert_eq!(u32::from_bytes(bytes), &[2, 3, 4, 5]);

    // Reading a leased buffer is fine; writing it is refused.
    let other = Buffer::<u32>::empty(&client, 4);
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&output).into(),
        (&other).into(),
    );
    assert_eq!(other.read(&client).unwrap(), vec![3, 4, 5, 6]);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        add_one::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            (&input).into(),
            (&output).into(),
        )
    }));
    assert!(refused.is_err());

    // Once the lease is dropped the buffer can be written again.
    drop(lease);
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&other).into(),
        (&output).into(),
    );
    assert_eq!(output.read(&client).unwrap(), vec![4, 5, 6, 7]);
}

/// Compute on the native runtime created on a wgpu device's Metal device,
/// lease the result, and read it through wgpu without a host copy in between.
#[cfg(feature = "wgpu")]
#[test]
fn a_lease_is_read_by_wgpu_on_the_same_device() {
    use cubecl_core::future::block_on;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("a Metal adapter");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        default_queue: wgpu::QueueDescriptor { label: None },
        ..Default::default()
    }))
    .unwrap();

    let metal = crate::wgpu_interop::from_wgpu(&device).expect("a Metal device behind wgpu");
    let client = R::client(&metal);
    let input = Buffer::create(&client, &[10u32, 20, 30, 40]);
    let output = Buffer::<u32>::empty(&client, 4);
    add_one::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(4),
        (&input).into(),
        (&output).into(),
    );
    let lease = block_on(client.export::<MetalServer>(output.handle())).unwrap();
    let (buffer, offset, size) = unsafe {
        crate::wgpu_interop::lease_as_wgpu(
            &device,
            &lease,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )
    };

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&buffer, offset, &staging, 0, size);
    queue.submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    assert_eq!(u32::from_bytes(&bytes), &[11, 21, 31, 41]);
    drop(lease);
}

/// The bridge refuses write usages and wraps its own device's leases for
/// wgpu to read.
#[cfg(feature = "wgpu")]
#[test]
fn the_bridge_checks_device_and_usage() {
    use cubecl_core::future::block_on;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("a Metal adapter");
    let (device, _queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let bridge = crate::wgpu_interop::WgpuBridge::new(&device).expect("Metal behind wgpu");
    let client = R::client(&bridge.metal());
    let output = Buffer::create(&client, &[1u32, 2, 3, 4]);
    let lease = block_on(client.export::<MetalServer>(output.handle())).unwrap();
    assert_eq!(
        bridge
            .buffer(
                &lease,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            )
            .err(),
        Some(crate::wgpu_interop::BridgeError::WriteUsage)
    );
    let (_, offset, size) = bridge.buffer(&lease, wgpu::BufferUsages::STORAGE).unwrap();
    assert_eq!(size, 16);
    assert!(offset % 4 == 0);
}
