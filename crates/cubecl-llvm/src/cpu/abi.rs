//! CPU kernel arguments.

use crate::{
    cpu::{
        entrypoint::InsertConstantEmulationPass, f16_evaluation::EvaluateF16Pass,
        unit_loops::UnitLoopsPass,
    },
    prelude::*,
    shared::metadata::{load_table, rebuild_func_type, table_ty},
};
use cubecl_runtime::config::compilation::F16Evaluation;

/// Buffers reach a CPU kernel through a table of pointers. Shared memory is the kernel's own
/// stack memory, made so by [`UnitLoopsPass`], and never reaches the table.
pub struct TableArgs;

impl EntryArgLayout for TableArgs {
    fn present_args(
        &self,
        ctx: &mut Context,
        func: FuncOp,
        buffers: &[(usize, usize, Value)],
        shared: SharedDeclarations,
    ) {
        assert!(
            shared.is_empty(),
            "CPU shared memory is lowered with the unit loops"
        );
        let entry = func.get_entry_block(ctx);

        if !buffers.is_empty() {
            let table_ty = table_ty(ctx);
            BasicBlock::insert_argument(entry, ctx, 0, table_ty);
            let buffer_ptrs = entry.deref(ctx).get_argument(0);
            let terminator = entry
                .deref(ctx)
                .get_terminator(ctx)
                .expect("entry block must be terminated");

            for (_idx, buffer_pos, old_val) in buffers.iter() {
                let buffer_ty = old_val.get_type(ctx);
                let buffer = load_table(ctx, buffer_ptrs, *buffer_pos, buffer_ty, terminator);
                old_val.replace_all_uses_with(ctx, &buffer);
            }

            let mut removed: Vec<usize> = buffers.iter().map(|(i, _, _)| i + 1).collect();
            removed.sort_unstable();
            for idx in removed.into_iter().rev() {
                BasicBlock::remove_argument(entry, ctx, idx);
            }
        }

        rebuild_func_type(ctx, func);
    }
}

pub struct CpuLowering {
    f16_evaluation: F16Evaluation,
}

impl CpuLowering {
    pub fn new(f16_evaluation: F16Evaluation) -> Self {
        Self { f16_evaluation }
    }
}

impl TargetLowering for CpuLowering {
    fn prologue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(InsertConstantEmulationPass);
    }

    fn lowered_control_flow(&self, passes: &mut OpPass<FuncOp, Passes>) {
        passes.add_pass(UnitLoopsPass);
    }

    /// F16 evaluation follows FMA contraction.
    fn epilogue(&self, passes: &mut OpPass<FuncOp, Passes>) {
        match self.f16_evaluation {
            F16Evaluation::PerOperation => {}
            F16Evaluation::Chain => passes.add_pass(EvaluateF16Pass {
                accumulators: false,
            }),
            F16Evaluation::Accumulators => passes.add_pass(EvaluateF16Pass { accumulators: true }),
        }
    }

    fn arg_layout(&self) -> Box<dyn EntryArgLayout> {
        Box::new(TableArgs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_args_is_a_boxed_layout() {
        let layout: Box<dyn EntryArgLayout> = Box::new(TableArgs);
        let _ = layout;
    }
}
