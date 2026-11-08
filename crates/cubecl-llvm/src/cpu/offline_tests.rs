//! CPU kernels lowered without a device, to the IR that is compiled ahead of time.

use crate::shared::offline_kernels::scale_kernel;
use crate::{PlironCompiler, PlironOptions};
use cubecl_core::ir::AddressType;
use cubecl_runtime::kernel::CubeKernel;

/// The IR names no target, so one lowering serves every CPU. With `CUBECL_CPU_IR_OUT` set, the
/// test writes the IR there.
#[test]
fn a_cpu_kernel_lowers_to_ir_without_a_target() {
    let ir = PlironCompiler::cpu_ir(
        scale_kernel(AddressType::U32).define(),
        &PlironOptions::default(),
    )
    .unwrap();
    assert!(
        ir.llvm.contains(&format!("@{}", ir.name)),
        "the entry point:\n{}",
        ir.llvm
    );
    assert!(
        !ir.llvm.contains("target triple"),
        "no target:\n{}",
        ir.llvm
    );
    if let Ok(path) = std::env::var("CUBECL_CPU_IR_OUT") {
        std::fs::write(path, &ir.llvm).unwrap();
    }
}
