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
    inst_combine::InstCombinePass, sccp::SCCPPass, simple_cse::SimpleCSEPass, sroa::SROAPass,
};

/// The first passes on a kernel function: split aggregates so their fields
/// can be checked and promoted on their own, then bound-check buffer accesses
/// according to the launch's `mode`.
pub fn add_kernel_entry_passes(passes: &mut Passes, mode: ExecutionMode, kernel_name: String) {
    passes.add_pass(SROAPass);
    passes.add_pass(CheckedIoPass::new(CheckedIo::new(mode, kernel_name)));
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
