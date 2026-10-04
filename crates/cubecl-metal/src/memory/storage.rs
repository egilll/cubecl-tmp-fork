use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;

/// An `MTLBuffer`, and the region of it a storage handle names (the whole
/// buffer as stored; the handle's own region once resolved). Send + Sync.
#[derive(Debug, Clone)]
pub struct MetalBufferHandle {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset: u64,
    size: u64,
}

// SAFETY: GPU memory access is synchronized via command buffer ordering.
unsafe impl Send for MetalBufferHandle {}
unsafe impl Sync for MetalBufferHandle {}

impl MetalBufferHandle {
    pub fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        let size = MTLBuffer::length(&*buffer) as u64;
        Self {
            buffer,
            offset: 0,
            size,
        }
    }

    pub fn inner(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }

    /// Where the region starts in the buffer, in bytes.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The region's length in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }
}

/// Metal buffer storage
#[derive(Debug)]
pub struct MetalStorage {
    buffers: HashMap<StorageId, MetalBufferHandle>,
    /// The bytes `buffers` holds.
    allocated: u64,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalStorage {
    pub fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            buffers: HashMap::new(),
            allocated: 0,
            device,
        }
    }
}

impl ComputeStorage for MetalStorage {
    type Resource = MetalBufferHandle;

    fn alignment(&self) -> usize {
        256 // Metal standard alignment
    }

    fn get(
        &mut self,
        handle: &StorageHandle,
    ) -> Result<Self::Resource, cubecl_core::server::IoError> {
        let buffer = self.buffers.get(&handle.id).ok_or_else(|| {
            cubecl_core::server::IoError::StorageHandleNotFound {
                reason: format!("{} in the Metal buffer storage", handle.id).into(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            }
        })?;
        Ok(MetalBufferHandle {
            buffer: buffer.buffer.clone(),
            offset: handle.offset(),
            size: handle.size(),
        })
    }

    fn alloc(&mut self, size: u64) -> Result<StorageHandle, cubecl_core::server::IoError> {
        use objc2_metal::MTLDevice;

        let id = StorageId::new();

        // MTLResourceStorageModeShared allows both CPU and GPU to access the buffer. Metal has no
        // empty buffer, so an empty one is backed by a byte: it still has to bind and copy.
        let buffer = (*self.device)
            .newBufferWithLength_options(
                size.max(1) as usize,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or_else(|| cubecl_core::server::IoError::Unknown {
                description: format!("Failed to allocate Metal buffer of size {}", size),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })?;

        // The device may round the length up; that is what it holds.
        self.allocated += objc2_metal::MTLBuffer::length(&*buffer) as u64;
        self.buffers.insert(id, MetalBufferHandle::new(buffer));

        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    fn dealloc(&mut self, id: StorageId) {
        use objc2_metal::MTLBuffer;

        if let Some(buffer) = self.buffers.remove(&id) {
            self.allocated -= buffer.inner().length() as u64;
        }
    }

    fn flush(&mut self) {
        // No deferred deallocations to flush.
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}
