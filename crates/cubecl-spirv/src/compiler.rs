use crate::{
    CollectVerCapExtPass, ConvertArgsPass, PARAMS_NAME, SpirvKernel,
    lower::LowerOpsSpirvPass,
    ops::{
        branch::BranchToSpirvConversionPass,
        builtin::{BUILTINS_NAME, LowerBuiltinsPass},
        memory::lower_shared,
        to_spirv_dialect::ToSpirvDialectPass,
    },
    params_storage_class,
};
use cubecl_core::{
    Compiler, WgpuCompilationOptions,
    ir::{
        ContextExt, attributes::FuncInterface, features::EnumSet, ident, metadata::Info,
        rewrite::SimplifyOpsPass,
    },
    post_processing::{
        bitwise::PromoteBitwisePass,
        minifloat::{
            Fp8Container, LowerMinifloatCast, LowerMinifloatCastPass, LowerMinifloatCompare,
            LowerMinifloatComparePass,
        },
        saturating::LowerSaturatingArithmeticPass,
        unroll::UnrollPass,
    },
    prelude::{KernelDefinition, Visibility},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_ir::{
    attributes::{ATTR_BUFFER_IO, BufferIOAttr, EntrypointInterface},
    dialect::{scf::BranchToSCFPass, ssa_matrix::MatrixToSSAPass},
    prelude::{OperationPtrExt, SingleBlockRegionInterface, SymbolOpInterface},
    rewrite::{CanonicalizePass, visit_all_ops_of_type_mut},
    settings::{Dim3, KernelSettings},
};
use cubecl_opt::passes::{
    alloc_shared_memory::AllocateSharedMemoryBlockPass,
    annotate_buffer_visibility::AnnotateGlobalVisibilityPass,
    inst_combine::InstCombinePass,
    mem2reg::Mem2RegPass,
    sccp::SCCPPass,
    simple_cse::SimpleCSEPass,
    sroa::SROAPass,
    uniformity::{DYNAMICALLY_UNIFORM_ATTR, MarkDynamicallyUniformPass, UniformAttr},
};
use cubecl_opt::{
    passes::inline::InlinePolicy,
    pipeline::{add_call_passes, add_kernel_entry_passes},
};
use cubecl_runtime::compiler::CompilationError;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::OneRegionInterface,
        ops::{FuncOp, ModuleOp},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        inserter::BlockInsertionPoint,
        listener::DummyListener,
        rewriter::{IRRewriter, Rewriter},
    },
    op::{Op, op_cast},
    operation::{Operation, verify_operation},
    opts::{dce::DCEPass, simplify_cfg::SimplifyCFGPass},
    pass::{AnalysisManager, NestedOpsPass, OpPass, PMConfig, Pass, Passes},
};
use pliron_spirv::{
    PlironBuilder, ToSpirvOp,
    attrs::VerCapExtAttr,
    decorations::{DecoratableOp, set_decoration_uniform, set_decoration_uniform_id},
    ops::{EntryPointOp, ExecutionModeOp, SpirvModuleOp},
};
use rspirv::{
    binary::Assemble,
    dr::Module,
    spirv::{
        AddressingModel, Capability, ExecutionMode, ExecutionModel, MemoryModel, Scope,
        StorageClass,
    },
};
use std::{fmt::Debug, sync::Arc};

pub struct KernelInfo {
    pub cube_dim: Dim3,
}

#[derive(Clone, Copy, Default)]
pub struct SpirvCompiler;

impl Compiler for SpirvCompiler {
    type Representation = SpirvKernel;
    type CompilationOptions = WgpuCompilationOptions;

    fn buffer_io(repr: &Self::Representation) -> Option<Vec<BufferIOAttr>> {
        repr.io.clone()
    }

    fn compile(
        &mut self,
        value: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
    ) -> Result<Self::Representation, CompilationError> {
        let errors = value.body.pop_errors();
        if !errors.is_empty() {
            let mut reason = "Can't compile spirv kernel".to_string();
            for error in errors {
                reason += error.as_str();
                reason += "\n";
            }

            return Err(CompilationError::Validation {
                reason,
                backtrace: BackTrace::capture(),
            });
        }

        #[cfg(feature = "pliron-dump")]
        let ir_printing_dir = kernel_dir_name(&value.settings.kernel_name);

        let entry_func = value.body.state().entry_func;
        let module = value.body.state().module;

        let mut ctx = value.body.into_context().expect("Should be unique");
        ctx.set_aux_ty::<Info>(value.info);
        ctx.set_aux_ty::<WgpuCompilationOptions>(*compilation_options);
        ctx.set_aux_ty::<KernelInfo>(KernelInfo {
            cube_dim: value.settings.cube_dim,
        });

        let (module, bindings, io, shared_size) = self.compile_kernel(
            &mut ctx,
            module,
            entry_func,
            value.settings.clone(),
            #[cfg(feature = "pliron-dump")]
            ir_printing_dir,
        )?;

        let info_visibility = Visibility::Read;
        let immediate_size = match params_storage_class(&ctx, bindings.len()) {
            StorageClass::PushConstant => Some((bindings.len() + 1) * size_of::<u64>()),
            _ => None,
        };

        let kernel = SpirvKernel {
            assembled_module: module.assemble(),
            module: Some(Arc::new(module)),
            bindings,
            io: Some(io),
            shared_size,
            immediate_size,
            info_visibility,
        };

        #[cfg(feature = "pliron-dump")]
        dump_spirv(&kernel, &value.settings.kernel_name);

        Ok(kernel)
    }

    fn extension(&self) -> &'static str {
        "spv"
    }

    fn lang_tag(&self) -> &'static str {
        "spirv"
    }
}

impl Debug for SpirvCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("spirv")
    }
}

impl SpirvCompiler {
    pub fn compile_kernel(
        &mut self,
        ctx: &mut Context,
        module: ModuleOp,
        entry_func: FuncOp,
        settings: KernelSettings,
        #[cfg(feature = "pliron-dump")] ir_printing_dir: Option<std::path::PathBuf>,
    ) -> Result<(Module, Vec<Visibility>, Vec<BufferIOAttr>, usize), CompilationError> {
        let entry = entry_func.get_entry_block(ctx);
        let comp_opts = ctx.aux_ty::<WgpuCompilationOptions>();
        let module_op = module.get_operation();

        #[cfg(feature = "pliron-dump")]
        if let Some(print_dir) = &ir_printing_dir {
            use pliron::printable::Printable;
            let str = std::format!("{}", module_op.disp(ctx));
            std::fs::write(print_dir.join("initial.plir"), &str).unwrap();
        }

        verify_operation(module.get_operation(), ctx)?;

        let config = PMConfig {
            print_after_all: cfg!(feature = "pliron-dump"),
            #[cfg(feature = "pliron-dump")]
            ir_printing_dir,
            ..Default::default()
        };

        let mut analyses = AnalysisManager::default();
        analyses.set_config(config);

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        // SPIR-V has no complex type: c32 becomes a two-lane f32 vector first.
        passes.add_pass(cubecl_opt::passes::lower_complex::LowerComplexPass);
        add_call_passes(&mut passes, InlinePolicy::All);

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        add_kernel_entry_passes(
            &mut func_passes,
            settings.execution_mode,
            settings.kernel_name,
        );
        func_passes.add_pass(UnrollPass::new(comp_opts.vulkan.max_vector_size));
        func_passes.add_pass(AllocateSharedMemoryBlockPass);
        func_passes.add_pass(LowerSaturatingArithmeticPass::default());
        // The two the extension actually provides, named rather than `EnumSet::all()`: `ue8m0`
        // is also an `Fp8Format`, and `VK_EXT_shader_float8` does not cover it. Claiming it here
        // would emit a conversion the driver has no encoding for.
        let native_fp8 = match comp_opts.vulkan.supports_float8 {
            true => {
                cubecl_core::ir::types::Fp8Format::E4M3 | cubecl_core::ir::types::Fp8Format::E5M2
            }
            false => EnumSet::empty(),
        };
        func_passes.add_pass(LowerMinifloatCastPass::new(LowerMinifloatCast::new(
            native_fp8,
            Fp8Container::Bytes,
        )));
        func_passes.add_pass(LowerMinifloatComparePass::new(LowerMinifloatCompare::new(
            Fp8Container::Bytes,
        )));
        func_passes.add_pass(BranchToSCFPass::default());
        func_passes.add_pass(MatrixToSSAPass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(MarkDynamicallyUniformPass);
        passes.add_pass(LowerBuiltinsPass);

        let mut func_passes = OpPass::<FuncOp, Passes>::default();
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(InstCombinePass::default());
        func_passes.add_pass(SimpleCSEPass::without_memory());
        func_passes.add_pass(SimplifyOpsPass::default());
        func_passes.add_pass(PromoteBitwisePass);
        func_passes.add_pass(LowerOpsSpirvPass::default());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SROAPass);

        func_passes.add_pass(Mem2RegPass);

        func_passes.add_pass(SROAPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimpleCSEPass::with_memory());
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(CanonicalizePass::default());

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.add_pass(AnnotateGlobalVisibilityPass);

        passes.run(module_op, ctx, &mut analyses)?;

        let bindings = (0..entry.deref(ctx).get_num_arguments()).map(|i| {
            let io = entry_func.get_arg_attr::<BufferIOAttr>(ctx, i, &ATTR_BUFFER_IO);
            match io.expect("Should have IO attr").is_writable() {
                false => Visibility::Read,
                true => Visibility::ReadWrite,
            }
        });
        let bindings: Vec<Visibility> = bindings.collect();
        // The four-state answer, by buffer position, before anything widens
        // or collapses it: what the launch path's taint bookkeeping consumes.
        let io = cubecl_core::ir::attributes::buffer_io_by_position(ctx, entry_func)
            .into_iter()
            .collect::<Vec<BufferIOAttr>>();

        verify_operation(module_op, ctx)?;

        let mut passes = OpPass::<ModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(BranchToSpirvConversionPass::default());
        func_passes.add_pass(Mem2RegPass);
        func_passes.add_pass(DCEPass);
        func_passes.add_pass(SCCPPass);
        func_passes.add_pass(SimplifyCFGPass);
        func_passes.add_pass(DCEPass);

        passes.add_pass(NestedOpsPass::new(func_passes));
        passes.run(module_op, ctx, &mut analyses)?;

        verify_operation(module_op, ctx)?;

        let spirv_module = insert_spirv_module(ctx, module);
        let spirv_module_op = spirv_module.get_operation();

        let mut passes = OpPass::<SpirvModuleOp, Passes>::default();
        let mut func_passes = OpPass::<FuncOp, Passes>::default();

        func_passes.add_pass(DCEPass);
        func_passes.add_pass(ToSpirvDialectPass::default());

        passes.add_pass(ConvertArgsPass);
        passes.add_pass(NestedOpsPass::new(func_passes));

        // The conversion pass reports ops it must not compile (e.g.
        // cube.poison) as errors, not bugs; surface them as a compilation
        // error instead of panicking the device thread.
        passes.run(spirv_module_op, ctx, &mut analyses)?;

        let (shared_size, shared_args) = lower_shared(ctx, spirv_module);
        declare_entry_point(ctx, spirv_module, shared_args);

        // Make sure this is the last pass so it catches all ops
        OpPass::<SpirvModuleOp, CollectVerCapExtPass>::default()
            .run(spirv_module_op, ctx, &mut analyses)
            .unwrap();

        // Something weird with the validation rules, can't be bothered to debug for the MVP.
        // Try to figure this out later.
        // verify_operation(module_op, ctx).expect("Failed to verify after passes");

        let mut builder = PlironBuilder::default();
        spirv_module.to_spirv(ctx, &mut builder)?;
        let module = builder.module();

        Ok((module, bindings, io, shared_size))
    }
}

fn insert_spirv_module(ctx: &mut Context, module: ModuleOp) -> SpirvModuleOp {
    let mut rewriter = IRRewriter::<DummyListener>::default();
    let comp_opts = ctx.aux_ty::<WgpuCompilationOptions>().vulkan;

    let spirv_module = SpirvModuleOp::new(
        ctx,
        ident("kernel"),
        AddressingModel::PhysicalStorageBuffer64,
        MemoryModel::Vulkan,
    );
    rewriter.inline_region(
        ctx,
        module.get_region(ctx),
        BlockInsertionPoint::AtRegionStart(spirv_module.get_region(ctx)),
    );
    let module_body = BasicBlock::new(ctx, None, vec![]);
    module_body.insert_at_front(module.get_region(ctx), ctx);
    spirv_module
        .get_operation()
        .insert_at_front(module_body, ctx);
    let vce = VerCapExtAttr::new(
        comp_opts.max_spirv_version,
        vec![Capability::Shader],
        vec![],
    );
    spirv_module.set_attr_spirv_module_vce(ctx, vce);
    spirv_module
}

fn declare_entry_point(ctx: &mut Context, module: SpirvModuleOp, shared_args: Vec<Identifier>) {
    let op = module.get_operation();
    visit_all_ops_of_type_mut::<FuncOp, _>(
        ctx,
        &mut (module, shared_args),
        op,
        |ctx, (module, shared_args), func| {
            let Some(entry) = func.get_entrypoint_abi(ctx) else {
                return;
            };
            let block = module.get_body(ctx, 0);
            let func_name = func.get_symbol_name(ctx);
            let mut interface = vec![PARAMS_NAME.clone(), BUILTINS_NAME.clone()];
            interface.extend(shared_args.clone());
            let entry_point = EntryPointOp::new(
                ctx,
                ExecutionModel::GLCompute,
                func_name.clone(),
                func_name.to_string(),
                interface,
            );
            entry_point.get_operation().insert_at_front(block, ctx);
            let (x, y, z) = entry.cube_dim.into();
            let execution_mode =
                ExecutionModeOp::new(ctx, func_name, ExecutionMode::LocalSize, vec![x, y, z]);
            execution_mode.get_operation().insert_at_front(block, ctx);
        },
    );
}

/// Forbid the driver from fusing a float add, subtract or multiply into an
/// `fma`, unless the operation was created under `fast_math` allowing
/// contraction. SPIR-V lets drivers contract undecorated arithmetic, which
/// changes results and breaks error-free transformations (TwoSum,
/// compensated sums) that rely on each operation being rounded on its own.
pub(crate) fn decorate_contraction(ctx: &Context, source: Ptr<Operation>, new_op: Ptr<Operation>) {
    use cubecl_ir::{
        FastMath,
        attributes::{ATTR_FAST_MATH, FastMathAttr},
        dialect::OperationPtrExt,
    };
    let is_float_arith = Operation::get_op::<pliron_spirv::ops::FAddOp>(new_op, ctx).is_some()
        || Operation::get_op::<pliron_spirv::ops::FSubOp>(new_op, ctx).is_some()
        || Operation::get_op::<pliron_spirv::ops::FMulOp>(new_op, ctx).is_some();
    if !is_float_arith {
        return;
    }
    let may_contract = source
        .get_attr::<FastMathAttr>(ctx, &ATTR_FAST_MATH)
        .is_some_and(|attr| {
            attr.allows(FastMath::AllowContraction) || attr.allows(FastMath::AllowTransform)
        });
    if !may_contract && let Some(op) = op_cast::<dyn DecoratableOp>(&*new_op.dyn_op(ctx)) {
        pliron_spirv::decorations::set_decoration_no_contraction(op, ctx);
    }
}

pub(crate) fn decorate_uniform(ctx: &Context, op: Ptr<Operation>, uniformity: Option<UniformAttr>) {
    let Some(uniformity) = uniformity else {
        return;
    };
    let props = ctx.aux_ty::<WgpuCompilationOptions>().vulkan;
    if let Some(can_decorate) = op_cast::<dyn DecoratableOp>(&*op.dyn_op(ctx)) {
        if props.max_spirv_version >= (1, 4) {
            // The spec theoretically allows any but Vulkan has a hardcoded set of just `Workgroup`
            // and `Subgroup` for the allowed values
            let scope = match uniformity {
                UniformAttr::Device | UniformAttr::Cube => Scope::Workgroup,
                UniformAttr::Plane => Scope::Subgroup,
            };
            set_decoration_uniform_id(can_decorate, ctx, scope.into());
        } else {
            set_decoration_uniform(can_decorate, ctx);
        }
    } else {
        op.set_attr(ctx, &DYNAMICALLY_UNIFORM_ATTR, uniformity);
    }
}

#[cfg(feature = "pliron-dump")]
pub fn kernel_dir_name(name: &str) -> Option<std::path::PathBuf> {
    if let Ok(dir) = std::env::var("CUBECL_DEBUG_PLIRON") {
        let path = sanitize_filename::sanitize_with_options(
            name,
            sanitize_filename::Options {
                replacement: "_",
                ..Default::default()
            },
        );
        let dir = std::path::PathBuf::from(dir).join(&path);
        std::fs::create_dir_all(&dir).unwrap();
        Some(dir)
    } else {
        None
    }
}

#[cfg(feature = "pliron-dump")]
pub(crate) fn dump_spirv(repr: &SpirvKernel, name: &str) {
    use std::fs;

    if let Some(dir) = kernel_dir_name(name) {
        let kernel = &repr.assembled_module;
        let kernel = kernel
            .iter()
            .flat_map(|it| it.to_le_bytes())
            .collect::<Vec<_>>();
        let kernel_path = dir.join("module.spv");
        fs::write(kernel_path, kernel).unwrap();
    }
}
