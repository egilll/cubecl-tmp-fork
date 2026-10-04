//! Completed buffers handed to another consumer of the same device.
//!
//! A [`Lease`] holds a buffer's backing resource, pinned: its allocation is
//! neither freed, reused nor relocated while the lease lives, and no launch
//! may write it. That is what lets a renderer read data CubeCL produced in
//! place, without a copy, on its own queue: the data is complete when the
//! lease is handed out and immutable until the lease is dropped.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;
use core::sync::atomic::{AtomicU64, Ordering};

use cubecl_environment::sync::Mutex;

use crate::memory_management::ManagedMemoryId;
use crate::server::BufferBinding;
use crate::storage::ManagedResource;

/// The regions under a live lease on one device.
#[derive(Debug, Default)]
pub struct Leases {
    next: AtomicU64,
    live: Mutex<Vec<(u64, ManagedMemoryId, Range<u64>)>>,
}

impl Leases {
    pub(crate) fn add(&self, binding: &BufferBinding) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (memory, start, end) = binding.claim_key();
        self.live.lock().push((id, memory, start..end));
        id
    }

    fn remove(&self, id: u64) {
        self.live.lock().retain(|(lease, _, _)| *lease != id);
    }

    /// Whether any live lease covers bytes of `binding`.
    pub fn covers(&self, binding: &BufferBinding) -> bool {
        let (memory, start, end) = binding.claim_key();
        self.live
            .lock()
            .iter()
            .any(|(_, leased, range)| *leased == memory && range.start < end && start < range.end)
    }
}

/// A completed buffer's backing resource, pinned for another consumer of
/// the device; see the [module documentation](self).
///
/// Drop it once that consumer's own use of the resource completed (for a
/// wgpu renderer, from `Queue::on_submitted_work_done`).
pub struct Lease<R: Send> {
    resource: ManagedResource<R>,
    leases: Arc<Leases>,
    id: u64,
}

impl<R: Send> Lease<R> {
    pub(crate) fn new(resource: ManagedResource<R>, leases: Arc<Leases>, id: u64) -> Self {
        Self {
            resource,
            leases,
            id,
        }
    }

    /// The backing resource. It may be larger than the leased buffer (a
    /// page holding several allocations); the backend's resource type says
    /// where the buffer starts in it and how long it is.
    pub fn resource(&self) -> &R {
        self.resource.resource()
    }
}

impl<R: Send> Drop for Lease<R> {
    fn drop(&mut self) {
        self.leases.remove(self.id);
    }
}

impl<R: Send> core::fmt::Debug for Lease<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Lease").field("id", &self.id).finish()
    }
}
