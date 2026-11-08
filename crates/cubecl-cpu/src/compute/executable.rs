//! A kernel as the CPU runs it: an entry point with a fixed ABI, compiled by the JIT at run time or
//! ahead of time and linked into the binary.

use std::{any::Any, ffi::c_void, ops::Range, sync::Arc};

use cubecl_server::kernel::BufferIOAttr;

/// The entry point of a CPU kernel: the buffer pointers, the cube count x/y/z, the linear range of
/// cubes the call runs, the range of units of each cube it runs, and the metadata.
pub type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u64, u64, u32, u32, *mut u64);

/// A kernel ready to run.
#[derive(Clone)]
pub struct Executable {
    entry: KernelFn,
    whole_cubes: bool,
    io: Arc<[BufferIOAttr]>,
    /// What keeps `entry` valid: the JIT that compiled it. Linked kernels need nothing.
    _owner: Option<Arc<dyn Any + Send + Sync>>,
}

impl Executable {
    /// A kernel at `entry`. `whole_cubes`: the units of a cube meet at barriers or share memory, so
    /// each call must run whole cubes. `io`: what it does with each buffer, in binding order.
    pub fn new(
        entry: KernelFn,
        whole_cubes: bool,
        io: Arc<[BufferIOAttr]>,
        owner: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Self {
        Self {
            entry,
            whole_cubes,
            io,
            _owner: owner,
        }
    }

    /// Whether each call must run whole cubes.
    pub fn whole_cubes(&self) -> bool {
        self.whole_cubes
    }

    /// Buffer access modes in binding order.
    pub fn io(&self) -> &[BufferIOAttr] {
        &self.io
    }

    /// Runs one call of a launch.
    pub(crate) fn run(&self, call: &Call) {
        let [count_x, count_y, count_z] = call.cube_count;
        (self.entry)(
            call.shared.buffer_ptrs.as_ptr() as *mut *mut c_void,
            count_x,
            count_y,
            count_z,
            call.cubes.start,
            call.cubes.end,
            call.units.start,
            call.units.end,
            call.shared.metadata.as_ptr() as *mut u64,
        );
    }
}

/// What every call of a launch shares.
pub(crate) struct Launch {
    pub buffer_ptrs: Vec<*mut c_void>,
    pub metadata: Vec<u64>,
    /// Keeps buffer storage alive until the launch completes.
    pub _keepalive: Vec<Box<dyn Any + Send>>,
}

// SAFETY: buffer storage outlives the launch, and `_keepalive` is only dropped, never accessed.
unsafe impl Send for Launch {}
unsafe impl Sync for Launch {}

/// One call of a launch: the cubes it runs, and which of their units.
#[derive(Clone)]
pub(crate) struct Call {
    pub shared: Arc<Launch>,
    pub cube_count: [u32; 3],
    /// Cubes by linear index, x fastest.
    pub cubes: Range<u64>,
    /// Units by index within their cube.
    pub units: Range<u32>,
}
