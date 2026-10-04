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

/// Why a lease could not cross to wgpu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeError {
    /// The lease belongs to another Metal device.
    ForeignLease,
    /// wgpu may only read a lease: it stays immutable until dropped.
    WriteUsage,
}

/// The native runtime behind one wgpu device, and the way back: leases of
/// its buffers cross to that wgpu device safely, because the bridge checks
/// where each lease comes from and that wgpu will only read it.
#[derive(Clone)]
pub struct WgpuBridge {
    device: wgpu::Device,
    raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    metal: MetalDevice,
}

impl core::fmt::Debug for WgpuBridge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuBridge").field("metal", &self.metal).finish()
    }
}

impl WgpuBridge {
    /// The bridge of `device`, or `None` when it doesn't run on wgpu's Metal
    /// backend.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        // SAFETY: the raw device is only retained, never destroyed here.
        let raw = unsafe { device.as_hal::<wgpu_hal::api::Metal>() }?
            .raw_device()
            .clone();
        Some(Self {
            device: device.clone(),
            metal: register_device(raw.clone()),
            raw,
        })
    }

    /// The native Metal device to create clients on.
    pub fn metal(&self) -> MetalDevice {
        self.metal.clone()
    }

    /// `lease`'s buffer as a [`wgpu::Buffer`] with the offset and length of
    /// the leased bytes in it, as [`lease_as_wgpu`] gives, for read-only
    /// `usage`.
    ///
    /// # Errors
    ///
    /// When the lease is of another device's buffer, or `usage` would let
    /// wgpu write it.
    pub fn buffer(
        &self,
        lease: &cubecl_core::lease::Lease<MetalBufferHandle>,
        usage: wgpu::BufferUsages,
    ) -> Result<(wgpu::Buffer, u64, u64), BridgeError> {
        use objc2_metal::MTLResource;

        let read = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_SRC;
        if !read.contains(usage) {
            return Err(BridgeError::WriteUsage);
        }
        let owner = lease.resource().inner().device();
        if !core::ptr::eq(
            core::ptr::from_ref(&*owner).cast::<u8>(),
            core::ptr::from_ref(&*self.raw).cast::<u8>(),
        ) {
            return Err(BridgeError::ForeignLease);
        }
        // SAFETY: the lease's buffer is on this bridge's device, which is
        // `self.device`'s, and `usage` only reads it.
        Ok(unsafe { lease_as_wgpu(&self.device, lease, usage) })
    }
}
