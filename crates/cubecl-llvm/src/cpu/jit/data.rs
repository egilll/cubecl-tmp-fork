use std::{ffi::c_void, ops::Range, sync::Arc};

/// Resources shared by all calls of a launch.
#[derive(Default)]
pub struct SharedData {
    pub buffer_ptrs: Vec<*mut c_void>,
    pub metadata: Vec<u64>,
    /// Keeps buffer storage alive until the launch completes.
    pub keepalive: Vec<Box<dyn std::any::Any + Send>>,
}

/// SAFETY: Buffer storage outlives the launch, and `keepalive` contents are only dropped, never
/// accessed concurrently.
unsafe impl Send for SharedData {}
unsafe impl Sync for SharedData {}

/// The arguments of one call of a kernel: the cubes it runs, and which of their units.
#[derive(Clone, Default)]
pub struct PlironData {
    pub shared: Arc<SharedData>,
    pub cube_count: [u32; 3],
    /// Cubes by linear index, x fastest.
    pub cubes: Range<u64>,
    /// Units by index within their cube.
    pub units: Range<u32>,
}

impl PlironData {
    pub fn new(
        buffer_ptrs: Vec<*mut c_void>,
        metadata: Vec<u64>,
        cube_count: [u32; 3],
        keepalive: Vec<Box<dyn std::any::Any + Send>>,
    ) -> Self {
        Self {
            shared: Arc::new(SharedData {
                buffer_ptrs,
                metadata,
                keepalive,
            }),
            cube_count,
            cubes: 0..0,
            units: 0..0,
        }
    }

    /// The same launch, running `cubes` and, of each, `units`.
    pub fn call(&self, cubes: Range<u64>, units: Range<u32>) -> Self {
        Self {
            shared: self.shared.clone(),
            cube_count: self.cube_count,
            cubes,
            units,
        }
    }
}
