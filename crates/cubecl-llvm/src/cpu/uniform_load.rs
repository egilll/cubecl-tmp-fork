//! Loads every unit of a cube reads alike: a cube barrier, then the load.

use crate::prelude::*;
use cubecl_core::ir::dialect::{
    atomic::AtomicLoadOp,
    memory::LoadOp,
    plane::{AtomicUniformLoadOp, UniformLoadOp},
    synchronization::{SyncOp, SyncScope, SyncScopeAttr},
};

#[op_interface_impl]
impl LowerOp for UniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))]
    }
}

#[op_interface_impl]
impl LowerOp for AtomicUniformLoadOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        scope.register(&SyncOp::new(
            scope.ctx_mut(),
            SyncScopeAttr::new(SyncScope::Cube),
        ));
        let ptr = self.ptr(scope.ctx());
        vec![scope.register_with_result(&AtomicLoadOp::new(scope.ctx_mut(), ptr))]
    }
}
