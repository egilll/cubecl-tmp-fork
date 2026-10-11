//! The builtins of a CPU kernel, read as one unit of one cube.
//!
//! The kernel body is compiled for a single unit. Its cube and unit coordinates are left as
//! builtin reads, which [`UnitLoopsPass`](super::unit_loops::UnitLoopsPass) replaces once it has
//! wrapped the body in loops over the cubes and units a launch hands the function.

use crate::{
    prelude::*,
    shared::builtin_values::{
        BuiltinValues, Replacer, absolute_pos, absolute_pos_x, absolute_pos_y, absolute_pos_z,
        constant, cube_count, cube_pos, set_dim_and_cluster_constants, unit_pos,
    },
};
use cubecl_core::{ir::dialect::general::ReadBuiltinOp, prelude::*};
use pliron::{dict_key, linked_list::ContainsLinkedList};

dict_key!(
    /// The launch's cube count along x.
    ATTR_CUBE_COUNT_X, "cube_count_x"
);
dict_key!(
    /// The launch's cube count along y.
    ATTR_CUBE_COUNT_Y, "cube_count_y"
);
dict_key!(
    /// The first cube of the launch's linear range the call runs.
    ATTR_CUBE_BEGIN, "cube_begin"
);
dict_key!(
    /// One past the last cube the call runs.
    ATTR_CUBE_END, "cube_end"
);
dict_key!(
    /// The first unit of each cube the call runs.
    ATTR_UNIT_BEGIN, "unit_begin"
);
dict_key!(
    /// One past the last unit of each cube the call runs.
    ATTR_UNIT_END, "unit_end"
);

/// The coordinates [`UnitLoopsPass`](super::unit_loops::UnitLoopsPass) provides.
pub const LOOP_BUILTINS: [Builtin; 6] = [
    Builtin::CubePosX,
    Builtin::CubePosY,
    Builtin::CubePosZ,
    Builtin::UnitPosX,
    Builtin::UnitPosY,
    Builtin::UnitPosZ,
];

/// Arguments every CPU kernel takes after its buffers: the cube count, then the ranges of cubes
/// and units the call runs.
#[derive(Default)]
pub struct InsertConstantEmulationPass;

#[pass_name]
impl Pass for InsertConstantEmulationPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let Some(func) = op.as_op::<FuncOp>(ctx) else {
            return Ok(res);
        };
        let Some(abi) = func.get_entrypoint_abi(ctx) else {
            return Ok(res);
        };
        let cube_dim = abi.cube_dim;
        let cluster_dim = abi.cluster_dim.unwrap_or(Dim3::new_single());

        let entry_block = func.get_entry_block(ctx);
        let first = entry_block.deref(ctx).iter(ctx).next();
        let mut builtins = BuiltinValues::default();
        let mut placeholders = Vec::new();
        {
            let mut inserter = match first {
                Some(first) => OpInserter::new_before_operation(first),
                None => OpInserter::new_at_block_end(entry_block),
            };
            let scope = Scope::from_context_and_inserter(ctx, &mut inserter);

            let u32_ty = u32::__expand_as_type(&scope);
            let u64_ty = u64::__expand_as_type(&scope);
            let argument = |ty, key: Option<&Identifier>| {
                let arg = func.push_argument(scope.ctx(), ty);
                if let Some(key) = key {
                    func.set_arg_attr_unit(scope.ctx(), arg, key);
                }
                entry_block.deref(scope.ctx()).get_argument(arg)
            };
            builtins.set(
                Builtin::CubeCountX,
                argument(u32_ty, Some(&ATTR_CUBE_COUNT_X)),
            );
            builtins.set(
                Builtin::CubeCountY,
                argument(u32_ty, Some(&ATTR_CUBE_COUNT_Y)),
            );
            builtins.set(Builtin::CubeCountZ, argument(u32_ty, None));
            argument(u64_ty, Some(&ATTR_CUBE_BEGIN));
            argument(u64_ty, Some(&ATTR_CUBE_END));
            argument(u32_ty, Some(&ATTR_UNIT_BEGIN));
            argument(u32_ty, Some(&ATTR_UNIT_END));

            for builtin in LOOP_BUILTINS {
                let read = ReadBuiltinOp::new(scope.ctx_mut(), u32_ty, builtin);
                scope.register(&read);
                placeholders.push(read.get_operation());
                builtins.set(builtin, read.get_result(scope.ctx()));
            }

            derive_builtins(&scope, &mut builtins, cube_dim, cluster_dim);
        }

        let mut state = (
            Replacer {
                builtins: &builtins,
                replacements: Vec::new(),
            },
            placeholders,
        );
        visit_all_ops_of_type::<ReadBuiltinOp, _>(ctx, &mut state, op, |ctx, state, read| {
            let (replacer, placeholders) = state;
            if placeholders.contains(&read.get_operation()) {
                return;
            }
            if let Some(value) = replacer.builtins.get(read.builtin(ctx).0) {
                replacer.replacements.push((read.get_result(ctx), value));
            }
        });
        for (old_value, new_value) in state.0.replacements {
            old_value.replace_all_uses_with(ctx, &new_value);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

pub fn runtime_arg(ctx: &Context, func: FuncOp, key: &Identifier) -> Value {
    let entry = func.get_entry_block(ctx);
    let num_args = entry.deref(ctx).get_num_arguments();
    (0..num_args)
        .find(|idx| func.has_arg_attr(ctx, *idx, key))
        .map(|idx| entry.deref(ctx).get_argument(idx))
        .unwrap_or_else(|| panic!("entry point must carry the '{key}' argument"))
}

/// Every builtin the coordinates decide.
fn derive_builtins(scope: &Scope, builtins: &mut BuiltinValues, cube_dim: Dim3, cluster_dim: Dim3) {
    set_dim_and_cluster_constants(scope, builtins, cube_dim, cluster_dim);

    let mut set_const = |builtin: Builtin, value: u32| {
        builtins.set(builtin, constant::expand(scope, value).value(scope));
    };

    // CPU planes contain one unit.
    set_const(Builtin::PlaneDim, 1);
    set_const(Builtin::UnitPosPlane, 0);

    let unit_pos = unit_pos::expand(
        scope,
        builtins.expect(Builtin::UnitPosX).into(),
        builtins.expect(Builtin::UnitPosY).into(),
        builtins.expect(Builtin::UnitPosZ).into(),
        cube_dim.x,
        cube_dim.y,
    )
    .value(scope);
    builtins.set(Builtin::UnitPos, unit_pos);
    builtins.set(Builtin::PlanePos, unit_pos);

    let cube_count = cube_count::expand(
        scope,
        builtins.expect(Builtin::CubeCountX).into(),
        builtins.expect(Builtin::CubeCountY).into(),
        builtins.expect(Builtin::CubeCountZ).into(),
    )
    .value(scope);
    builtins.set(Builtin::CubeCount, cube_count);

    let absolute_pos_x = absolute_pos_x::expand(
        scope,
        builtins.expect(Builtin::CubePosX).into(),
        builtins.expect(Builtin::UnitPosX).into(),
        cube_dim.x,
    )
    .value(scope);
    builtins.set(Builtin::AbsolutePosX, absolute_pos_x);
    let absolute_pos_y = absolute_pos_y::expand(
        scope,
        builtins.expect(Builtin::CubePosY).into(),
        builtins.expect(Builtin::UnitPosY).into(),
        cube_dim.y,
    )
    .value(scope);
    builtins.set(Builtin::AbsolutePosY, absolute_pos_y);
    let absolute_pos_z = absolute_pos_z::expand(
        scope,
        builtins.expect(Builtin::CubePosZ).into(),
        builtins.expect(Builtin::UnitPosZ).into(),
        cube_dim.z,
    )
    .value(scope);
    builtins.set(Builtin::AbsolutePosZ, absolute_pos_z);

    let cube_pos = cube_pos::expand(
        scope,
        builtins.expect(Builtin::CubePosX).into(),
        builtins.expect(Builtin::CubePosY).into(),
        builtins.expect(Builtin::CubePosZ).into(),
        builtins.expect(Builtin::CubeCountX).into(),
        builtins.expect(Builtin::CubeCountY).into(),
    )
    .value(scope);
    builtins.set(Builtin::CubePos, cube_pos);

    let absolute_pos = absolute_pos::expand(
        scope,
        cube_pos.into(),
        unit_pos.into(),
        cube_dim.num_elems(),
    )
    .value(scope);
    builtins.set(Builtin::AbsolutePos, absolute_pos);
}
