use cubecl_core::CubeDim;

use crate::compute::executable::Executable;

/// A kernel compiled for the CPU, and the cube dimension it was compiled for.
#[derive(Clone)]
pub struct CpuKernel {
    pub(crate) executable: Executable,
    pub(crate) cube_dim: CubeDim,
    pub(crate) name: String,
}

impl core::fmt::Debug for CpuKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuKernel")
            .field("name", &self.name)
            .finish()
    }
}
