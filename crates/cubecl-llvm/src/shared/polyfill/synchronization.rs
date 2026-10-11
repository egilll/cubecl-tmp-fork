//! Target-specific synchronization.

use crate::prelude::*;
use cubecl_core::ir::dialect::synchronization::{SyncOp, SyncScope};

#[op_interface_impl]
impl LowerOp for SyncOp {
    /// A CPU cube barrier divides the unit loops instead.
    fn should_lower(&self, ctx: &Context) -> bool {
        !(ctx.target() == LlvmTarget::Cpu && self.scope(ctx).0 == SyncScope::Cube)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        let sync_scope = self.scope(ctx).0;
        let target = ctx.target();

        match sync_scope {
            SyncScope::Unit => {}
            SyncScope::Plane => match target {
                LlvmTarget::Cpu => {}
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_plane(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_plane(scope),
            },
            SyncScope::Cube => match target {
                LlvmTarget::Cpu => unreachable!("a CPU cube barrier divides the unit loops"),
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_cube(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_cube(scope),
            },
            // The units of the cube meet, as at a cube barrier, and what each wrote to storage is
            // visible to every cube of the device, as what other cubes published is visible to it.
            SyncScope::Device => match target {
                LlvmTarget::Cpu => {
                    panic!("Device wide synchronization is not supported by the CPU runtime")
                }
                #[cfg(feature = "amdgpu")]
                LlvmTarget::AmdGpu => crate::amdgpu::synchronization::lower_sync_storage(scope),
                #[cfg(feature = "nvptx")]
                LlvmTarget::Nvptx => crate::nvptx::synchronization::lower_sync_storage(scope),
            },
        }
        vec![]
    }
}
