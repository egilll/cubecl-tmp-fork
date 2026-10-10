//! Pass sequences that several targets share.
//!
//! Each target still builds its own pipeline, because its lowerings differ
//! and have to run at specific points. The sequences here are the parts that
//! don't differ, so a change to them (a new pass, a new order) is made once
//! rather than once per target.

use alloc::string::String;
use cubecl_core::{
    ir::{rewrite::SimplifyOpsPass, settings::ExecutionMode},
    post_processing::checked_io::{CheckedIo, CheckedIoPass},
};
use pliron::{
    opts::{dce::DCEPass, mem2reg::Mem2RegPass},
    pass::Passes,
};

use crate::passes::{
    inline::{InlinePass, InlinePolicy},
    inst_combine::InstCombinePass,
    sccp::SCCPPass,
    simple_cse::SimpleCSEPass,
    sroa::SROAPass,
};

/// The module passes that run before any per-function pass: device function
/// calls are inlined as `policy` says, so the per-function pipeline sees the
/// calls the target will emit and nothing else.
pub fn add_call_passes(passes: &mut Passes, policy: InlinePolicy) {
    passes.add_pass(InlinePass::new(policy));
}

/// [`add_call_passes`], preceded by the kernel entry passes on every
/// function, for targets that keep calls: a device function's bounds checks
/// read buffer lengths, which the inliner then turns into parameters, so
/// they have to exist before it runs. The per-function pipeline must not add
/// the entry passes again.
pub fn add_entry_and_call_passes(
    passes: &mut Passes,
    policy: InlinePolicy,
    mode: ExecutionMode,
    kernel_name: String,
) {
    use cubecl_ir::pliron::{
        builtin::ops::FuncOp,
        pass::{NestedOpsPass, OpPass},
    };

    let mut entry = OpPass::<FuncOp, Passes>::default();
    add_kernel_entry_passes(&mut entry, mode, kernel_name);
    passes.add_pass(NestedOpsPass::new(entry));
    add_call_passes(passes, policy);
}

/// The first passes on a kernel function: split aggregates so their fields
/// can be checked and promoted on their own, then bound-check buffer accesses
/// according to the launch's `mode`.
pub fn add_kernel_entry_passes(passes: &mut Passes, mode: ExecutionMode, kernel_name: String) {
    passes.add_pass(SROAPass);
    passes.add_pass(CheckedIoPass::new(CheckedIo::new(mode, kernel_name)));
}

/// Fold indices and split local aggregates while their allocation operations
/// still expose the scalar-replacement interfaces. Target declarations need
/// not retain those interfaces after lowering.
pub fn add_pre_lowering_passes(passes: &mut Passes) {
    passes.add_pass(SCCPPass);
    passes.add_pass(SimplifyOpsPass::default());
    passes.add_pass(SROAPass);
    passes.add_pass(DCEPass);
}

/// The cleanup a target runs once its lowerings are done, for targets that
/// emit structured source from the `branch` dialect: constant propagation,
/// simplification, CSE and dead code elimination, run on both sides of
/// promoting locals to values, because each unlocks the other.
pub fn add_structured_cleanup_passes(passes: &mut Passes) {
    passes.add_pass(SCCPPass);
    passes.add_pass(InstCombinePass::default());
    passes.add_pass(SimpleCSEPass::without_memory());
    passes.add_pass(SimplifyOpsPass::default());
    passes.add_pass(DCEPass);
    passes.add_pass(SROAPass);

    passes.add_pass(Mem2RegPass);

    passes.add_pass(SROAPass);
    passes.add_pass(SCCPPass);
    passes.add_pass(SimpleCSEPass::with_memory());
    passes.add_pass(SimplifyOpsPass::default());
    passes.add_pass(DCEPass);
}
