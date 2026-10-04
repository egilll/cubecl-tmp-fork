//! Native Metal compute beside a wgpu renderer on the same GPU.
//!
//! [`from_wgpu`] runs `cubecl-metal` on the `MTLDevice` behind a wgpu device,
//! so compute gets its own command queues (and fault reporting, and the
//! native runtime's cheaper launches) while drawing stays on wgpu's. Results
//! cross over without a copy: [`Client::export`](cubecl_core::prelude::Client)
//! leases a completed buffer, and [`lease_as_wgpu`] wraps the leased
//! `MTLBuffer` as a [`wgpu::Buffer`] on the same device.
//!
//! Both sides use `objc2-metal`, so the objects pass straight through.

use crate::{MetalDevice, memory::MetalBufferHandle, register_device};

/// The native Metal device behind `device`, or `None` when `device` doesn't
/// run on wgpu's Metal backend.
pub fn from_wgpu(device: &wgpu::Device) -> Option<MetalDevice> {
    // SAFETY: the raw device is only retained, never destroyed here.
    let hal = unsafe { device.as_hal::<wgpu_hal::api::Metal>() }?;
    Some(register_device(hal.raw_device().clone()))
}

/// A leased buffer as a [`wgpu::Buffer`] on `device`, with the offset and
/// length of the leased bytes in it.
///
/// The returned buffer is the whole `MTLBuffer` the lease points into, which
/// may hold other allocations: read only `offset..offset + size`. Keep the
/// lease alive until wgpu's own use of the buffer completed (for instance
/// from `Queue::on_submitted_work_done`); the lease is what keeps CubeCL from
/// writing or reusing the bytes.
///
/// # Safety
///
/// `device` must be the wgpu device the lease's runtime was created on (see
/// [`from_wgpu`]), and `usage` must only ask for what a storage buffer in
/// shared memory supports.
pub unsafe fn lease_as_wgpu(
    device: &wgpu::Device,
    lease: &cubecl_core::lease::Lease<MetalBufferHandle>,
    usage: wgpu::BufferUsages,
) -> (wgpu::Buffer, u64, u64) {
    use objc2_metal::MTLBuffer;

    let resource = lease.resource();
    let raw = resource.inner().clone();
    let size = raw.length() as u64;
    // SAFETY: `raw` comes from this device (the caller's contract), is
    // initialized (the leased work completed), and is never empty: Metal
    // backs an empty CubeCL buffer with a byte.
    let buffer = unsafe {
        let hal = wgpu_hal::metal::Device::buffer_from_raw(raw);
        device.create_buffer_from_hal::<wgpu_hal::api::Metal>(
            hal,
            &wgpu::BufferDescriptor {
                label: Some("cubecl lease"),
                size,
                usage,
                mapped_at_creation: false,
            },
        )
    };
    (buffer, resource.offset(), resource.size())
}
