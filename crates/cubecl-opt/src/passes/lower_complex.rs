//! Complex numbers for targets without them.
//!
//! The frontend builds `c32` arithmetic from ordinary float operations on
//! values of type `cube.c32`, plus a few complex-only operations. A target
//! that has no complex type (MSL, WGSL, SPIR-V) runs this pass first: every
//! `c32` becomes a two-lane `f32` vector (real, imaginary), which has the same
//! size and layout, so buffers keep their bytes. Addition, subtraction,
//! negation, loads, stores and selects then already mean the right thing on
//! the vector; multiplication, division, comparisons and the math functions
//! are rewritten into their component formulas.
//!
//! Only scalar `c32` is lowered. `c64` needs `f64`, which these targets
//! lack, and vectors of complex numbers are refused with an error.

use alloc::{format, string::String, vec::Vec};

use cubecl_core::{self as cubecl, prelude::*};
use cubecl_ir::{
    ConstantValue, ElemType, ExpandValue, FloatKind, Scope,
    attributes::ComplexAttr,
    dialect::{
        cmp::{FEqualOp, FNotEqualOp},
        general::{CastOp, ReadScalarOp},
        math::*,
    },
    interfaces::ConstantAttr,
    prelude::*,
    types::{
        ArrayType, PointerType, RuntimeArrayType, VectorType,
        aggregate::{CheckedPtrType, SliceType},
        scalar::{Complex32Type, Complex64Type, Float32Type},
    },
};
use pliron::{
    attribute::AttrObj,
    builtin::{
        ops::{ConstantOp, FuncOp},
        types::FunctionType,
    },
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    input_error_noloc,
    irbuild::inserter::IRInserter,
    irbuild::listener::DummyListener,
    printable::Printable,
};

type C2 = Vector<f32, Const<2>>;

/// Rewrites `c32` into two-lane `f32` vectors; see the module documentation.
pub struct LowerComplexPass;

#[derive(Clone, Copy, Debug)]
enum Rewrite {
    Mul,
    Div,
    Eq,
    Ne,
    Conj,
    Real,
    Imag,
    Abs,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Tanh,
    Powf,
    Constant(f64, f64),
    FromFloat,
    /// A launch scalar: read with the lowered type, from the same field.
    ReadScalar,
}

#[derive(Debug, thiserror::Error)]
enum ComplexError {
    #[error(
        "{0} isn't supported on this target: it has no complex type, and only scalar c32 is lowered"
    )]
    Unsupported(String),
}

#[pass_name]
impl Pass for LowerComplexPass {
    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> Result<PassResult> {
        let mut res = PassResult::default();

        let mut ops = Vec::new();
        let mut blocks = Vec::new();
        walk_op(
            ctx,
            &mut (&mut ops, &mut blocks),
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |_, (ops, blocks), node| match node {
                IRNode::Operation(op) => ops.push(op),
                IRNode::BasicBlock(block) => blocks.push(block),
                IRNode::Region(_) => {}
            },
        );

        // Is there anything to do?
        let mentions = |ctx: &Context, ty: TypeHandle| mentions_complex(ctx, ty);
        let any = ops.iter().any(|op| {
            op.deref(ctx)
                .results()
                .any(|value| mentions(ctx, value.get_type(ctx)))
        }) || blocks.iter().any(|block| {
            block
                .deref(ctx)
                .arguments()
                .any(|arg| mentions(ctx, arg.get_type(ctx)))
        });
        if !any {
            return Ok(res);
        }
        // Types made from `ElemType` from now on (the info struct's fields)
        // are lowered too.
        ctx.set_aux_ty(cubecl_ir::ComplexLowered);

        // What each operation means, read while the types still say complex.
        let mut rewrites = Vec::new();
        for &op in &ops {
            if let Some(rewrite) = classify(ctx, op)? {
                rewrites.push((op, rewrite));
            }
        }

        // Every value, then every function signature.
        for &op in &ops {
            let results: Vec<Value> = op.deref(ctx).results().collect();
            for value in results {
                if let Some(ty) = map_type(ctx, value.get_type(ctx))? {
                    value.set_type(ctx, ty);
                }
            }
        }
        for &block in &blocks {
            let args: Vec<Value> = block.deref(ctx).arguments().collect();
            for arg in args {
                if let Some(ty) = map_type(ctx, arg.get_type(ctx))? {
                    arg.set_type(ctx, ty);
                }
            }
        }
        for &op in &ops {
            if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
                retype_function(ctx, func)?;
            }
        }

        for (op, rewrite) in rewrites {
            rewrite_op(ctx, op, rewrite);
        }

        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

fn is_c32(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).is::<Complex32Type>()
}

/// Whether `ty` holds a complex number anywhere.
fn mentions_complex(ctx: &Context, ty: TypeHandle) -> bool {
    let inner = {
        let ty = ty.deref(ctx);
        if ty.is::<Complex32Type>() || ty.is::<Complex64Type>() {
            return true;
        }
        if let Some(t) = ty.downcast_ref::<PointerType>() {
            Some(t.inner)
        } else if let Some(t) = ty.downcast_ref::<ArrayType>() {
            Some(t.inner)
        } else if let Some(t) = ty.downcast_ref::<RuntimeArrayType>() {
            Some(t.inner)
        } else if let Some(t) = ty.downcast_ref::<VectorType>() {
            Some(t.inner)
        } else if let Some(t) = ty.downcast_ref::<SliceType>() {
            Some(t.base_ty)
        } else {
            ty.downcast_ref::<CheckedPtrType>().map(|t| t.base_ty)
        }
    };
    inner.is_some_and(|inner| mentions_complex(ctx, inner))
}

/// `ty` with every `c32` replaced by a two-lane `f32` vector, or `None` when
/// it holds no complex number.
fn map_type(ctx: &Context, ty: TypeHandle) -> Result<Option<TypeHandle>> {
    if !mentions_complex(ctx, ty) {
        return Ok(None);
    }
    let deref = ty.deref(ctx);
    if deref.is::<Complex32Type>() {
        drop(deref);
        return Ok(Some(c2_type(ctx)));
    }
    if deref.is::<Complex64Type>() {
        return Err(input_error_noloc!(ComplexError::Unsupported("c64".into())));
    }
    if let Some(&PointerType {
        inner,
        address_space,
    }) = deref.downcast_ref::<PointerType>()
    {
        drop(deref);
        let inner = map_type(ctx, inner)?.expect("mentions complex");
        return Ok(Some(PointerType::get(ctx, inner, address_space).into()));
    }
    if let Some(&ArrayType { inner, length }) = deref.downcast_ref::<ArrayType>() {
        drop(deref);
        let inner = map_type(ctx, inner)?.expect("mentions complex");
        return Ok(Some(ArrayType::get(ctx, inner, length).into()));
    }
    if let Some(&RuntimeArrayType { inner }) = deref.downcast_ref::<RuntimeArrayType>() {
        drop(deref);
        let inner = map_type(ctx, inner)?.expect("mentions complex");
        return Ok(Some(RuntimeArrayType::get(ctx, inner).into()));
    }
    if let Some(slice) = deref.downcast_ref::<SliceType>() {
        let base = slice.base_ty;
        drop(deref);
        let base = map_type(ctx, base)?.expect("mentions complex");
        return Ok(Some(SliceType::get(ctx, base).into()));
    }
    if let Some(checked) = deref.downcast_ref::<CheckedPtrType>() {
        let inner = checked.base_ty;
        drop(deref);
        let inner = map_type(ctx, inner)?.expect("mentions complex");
        return Ok(Some(CheckedPtrType::get(ctx, inner).into()));
    }
    let name = format!("{}", ty.disp(ctx));
    Err(input_error_noloc!(ComplexError::Unsupported(name)))
}

fn c2_type(ctx: &Context) -> TypeHandle {
    VectorType::get(ctx, Float32Type::get(ctx).into(), 2).into()
}

fn retype_function(ctx: &mut Context, func: FuncOp) -> Result<()> {
    let ty = func.get_type(ctx);
    let (params, results) = {
        let ty = ty.deref(ctx);
        let fn_ty = ty
            .downcast_ref::<FunctionType>()
            .expect("a function has a function type");
        (fn_ty.arg_types().to_vec(), fn_ty.res_types().to_vec())
    };
    let mut changed = false;
    let mut map_all = |ctx: &Context, types: Vec<TypeHandle>| -> Result<Vec<TypeHandle>> {
        types
            .into_iter()
            .map(|ty| {
                Ok(match map_type(ctx, ty)? {
                    Some(mapped) => {
                        changed = true;
                        mapped
                    }
                    None => ty,
                })
            })
            .collect()
    };
    let params = map_all(ctx, params)?;
    let results = map_all(ctx, results)?;
    if changed {
        let new_ty = FunctionType::get(ctx, params, results);
        let new_ty: TypeHandle = new_ty.into();
        func.set_attr_builtin_func_type(ctx, new_ty.into());
    }
    Ok(())
}

/// What `op` needs rewritten to, judged from its complex types.
fn classify(ctx: &Context, op: Ptr<Operation>) -> Result<Option<Rewrite>> {
    let operand_c32 = |i: usize| {
        let operation = op.deref(ctx);
        (i < operation.get_num_operands()) && is_c32(ctx, operation.get_operand(i).get_type(ctx))
    };
    let result_c32 = op
        .deref(ctx)
        .results()
        .next()
        .is_some_and(|value| is_c32(ctx, value.get_type(ctx)));
    let is = |f: fn(Ptr<Operation>, &Context) -> bool| f(op, ctx);

    // Vectors of complex numbers aren't lowered: refuse them by name.
    let mentions_vector = op.deref(ctx).results().any(|value| {
        let ty = value.get_type(ctx);
        ty.deref(ctx)
            .downcast_ref::<VectorType>()
            .is_some_and(|vector| is_c32(ctx, vector.inner))
    });
    if mentions_vector {
        return Err(input_error_noloc!(ComplexError::Unsupported(
            "a vector of complex numbers".into()
        )));
    }

    macro_rules! of {
        ($ty:ty) => {
            |op: Ptr<Operation>, ctx: &Context| Operation::get_op::<$ty>(op, ctx).is_some()
        };
    }

    Ok(if is(of!(ReadScalarOp)) && result_c32 {
        Some(Rewrite::ReadScalar)
    } else if is(of!(ConstantOp)) && result_c32 {
        let constant = Operation::get_op::<ConstantOp>(op, ctx).unwrap();
        let value: AttrObj = constant
            .get_attr_builtin_constant_value(ctx)
            .expect("a constant has a value")
            .clone();
        let complex = value
            .downcast_ref::<ComplexAttr>()
            .expect("a complex constant holds a complex attribute");
        match complex.as_const_val(ctx) {
            ConstantValue::Complex(re, im) => Some(Rewrite::Constant(re, im)),
            _ => unreachable!("a complex attribute holds a complex value"),
        }
    } else if is(of!(CastOp)) && result_c32 && !operand_c32(0) {
        Some(Rewrite::FromFloat)
    } else if !operand_c32(0) {
        None
    } else if is(of!(FMulOp)) {
        Some(Rewrite::Mul)
    } else if is(of!(FDivOp)) {
        Some(Rewrite::Div)
    } else if is(of!(FEqualOp)) {
        Some(Rewrite::Eq)
    } else if is(of!(FNotEqualOp)) {
        Some(Rewrite::Ne)
    } else if is(of!(CConjOp)) {
        Some(Rewrite::Conj)
    } else if is(of!(CRealOp)) {
        Some(Rewrite::Real)
    } else if is(of!(CImagOp)) {
        Some(Rewrite::Imag)
    } else if is(of!(CAbsOp)) {
        Some(Rewrite::Abs)
    } else if is(of!(ExpOp)) {
        Some(Rewrite::Exp)
    } else if is(of!(LogOp)) {
        Some(Rewrite::Log)
    } else if is(of!(SinOp)) {
        Some(Rewrite::Sin)
    } else if is(of!(CosOp)) {
        Some(Rewrite::Cos)
    } else if is(of!(SqrtOp)) {
        Some(Rewrite::Sqrt)
    } else if is(of!(TanhOp)) {
        Some(Rewrite::Tanh)
    } else if is(of!(PowfOp)) {
        Some(Rewrite::Powf)
    } else {
        None
    })
}

fn rewrite_op(ctx: &mut Context, op: Ptr<Operation>, rewrite: Rewrite) {
    if let Rewrite::ReadScalar = rewrite {
        let read = Operation::get_op::<ReadScalarOp>(op, ctx).unwrap();
        let id = *read.id(ctx);
        let ty = c2_type(ctx);
        let new_op = ReadScalarOp::new(ctx, pliron::builtin::attributes::TypeAttr::new(ty), id);
        new_op.get_operation().insert_before(ctx, op);
        let new_value = new_op.get_result(ctx);
        let old = op.deref(ctx).get_result(0);
        old.replace_all_uses_with(ctx, &new_value);
        Operation::erase(op, ctx);
        return;
    }
    let operand =
        |ctx: &Context, i: usize| -> NativeExpand<C2> { op.deref(ctx).get_operand(i).into() };
    let lhs = (op.deref(ctx).get_num_operands() > 0).then(|| operand(ctx, 0));
    let rhs = (op.deref(ctx).get_num_operands() > 1).then(|| operand(ctx, 1));
    let float_operand: Option<NativeExpand<f32>> = match rewrite {
        Rewrite::FromFloat => Some(op.deref(ctx).get_operand(0).into()),
        _ => None,
    };

    let mut inserter = IRInserter::<DummyListener>::new_before_operation(op);
    let new_value = {
        let scope = Scope::from_context_and_inserter(ctx, &mut inserter);
        let scope = &scope;
        let a = || lhs.clone().expect("an operand");
        let b = || rhs.clone().expect("a second operand");
        match rewrite {
            Rewrite::Mul => c_mul::expand(scope, a(), b()).expand.read_value(scope),
            Rewrite::Div => c_div::expand(scope, a(), b()).expand.read_value(scope),
            Rewrite::Eq => c_eq::expand(scope, a(), b()).expand.read_value(scope),
            Rewrite::Ne => c_ne::expand(scope, a(), b()).expand.read_value(scope),
            Rewrite::Conj => c_conj::expand(scope, a()).expand.read_value(scope),
            Rewrite::Real => c_real::expand(scope, a()).expand.read_value(scope),
            Rewrite::Imag => c_imag::expand(scope, a()).expand.read_value(scope),
            Rewrite::Abs => c_abs::expand(scope, a()).expand.read_value(scope),
            Rewrite::Exp => c_exp::expand(scope, a()).expand.read_value(scope),
            Rewrite::Log => c_log::expand(scope, a()).expand.read_value(scope),
            Rewrite::Sin => c_sin::expand(scope, a()).expand.read_value(scope),
            Rewrite::Cos => c_cos::expand(scope, a()).expand.read_value(scope),
            Rewrite::Sqrt => c_sqrt::expand(scope, a()).expand.read_value(scope),
            Rewrite::Tanh => c_tanh::expand(scope, a()).expand.read_value(scope),
            Rewrite::Powf => c_pow::expand(scope, a(), b()).expand.read_value(scope),
            Rewrite::Constant(re, im) => {
                let float = |v: f64| -> NativeExpand<f32> {
                    ExpandValue::constant(ConstantValue::Float(v), ElemType::Float(FloatKind::F32))
                        .into()
                };
                c_make::expand(scope, float(re), float(im))
                    .expand
                    .read_value(scope)
            }
            Rewrite::FromFloat => {
                let zero: NativeExpand<f32> = ExpandValue::constant(
                    ConstantValue::Float(0.0),
                    ElemType::Float(FloatKind::F32),
                )
                .into();
                c_make::expand(scope, float_operand.clone().unwrap(), zero)
                    .expand
                    .read_value(scope)
            }
            Rewrite::ReadScalar => unreachable!("handled above"),
        }
    };
    let old = op.deref(ctx).get_result(0);
    old.replace_all_uses_with(ctx, &new_value);
    Operation::erase(op, ctx);
}

#[cube(inline)]
fn c_make(re: f32, im: f32) -> C2 {
    let mut v = C2::new(re);
    v.insert(1usize, im);
    v
}

#[cube(inline)]
fn c_mul(a: C2, b: C2) -> C2 {
    let (ar, ai, br, bi) = (
        a.extract(0usize),
        a.extract(1usize),
        b.extract(0usize),
        b.extract(1usize),
    );
    c_make(ar * br - ai * bi, ar * bi + ai * br)
}

#[cube(inline)]
fn c_div(a: C2, b: C2) -> C2 {
    let (ar, ai, br, bi) = (
        a.extract(0usize),
        a.extract(1usize),
        b.extract(0usize),
        b.extract(1usize),
    );
    let d = br * br + bi * bi;
    c_make((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)
}

#[cube(inline)]
fn c_eq(a: C2, b: C2) -> bool {
    let re = a.extract(0usize) == b.extract(0usize);
    let im = a.extract(1usize) == b.extract(1usize);
    select(re, im, false)
}

#[cube(inline)]
fn c_ne(a: C2, b: C2) -> bool {
    let re = a.extract(0usize) != b.extract(0usize);
    let im = a.extract(1usize) != b.extract(1usize);
    select(re, true, im)
}

#[cube(inline)]
fn c_conj(a: C2) -> C2 {
    c_make(a.extract(0usize), -a.extract(1usize))
}

#[cube(inline)]
fn c_real(a: C2) -> f32 {
    a.extract(0usize)
}

#[cube(inline)]
fn c_imag(a: C2) -> f32 {
    a.extract(1usize)
}

#[cube(inline)]
fn c_abs(a: C2) -> f32 {
    f32::hypot(a.extract(0usize), a.extract(1usize))
}

#[cube(inline)]
fn c_exp(a: C2) -> C2 {
    let (x, y) = (a.extract(0usize), a.extract(1usize));
    let ex = f32::exp(x);
    c_make(ex * f32::cos(y), ex * f32::sin(y))
}

#[cube(inline)]
fn c_log(a: C2) -> C2 {
    let (x, y) = (a.extract(0usize), a.extract(1usize));
    c_make(f32::ln(f32::hypot(x, y)), f32::atan2(y, x))
}

#[cube(inline)]
fn c_sin(a: C2) -> C2 {
    let (x, y) = (a.extract(0usize), a.extract(1usize));
    c_make(f32::sin(x) * f32::cosh(y), f32::cos(x) * f32::sinh(y))
}

#[cube(inline)]
fn c_cos(a: C2) -> C2 {
    let (x, y) = (a.extract(0usize), a.extract(1usize));
    c_make(f32::cos(x) * f32::cosh(y), -(f32::sin(x) * f32::sinh(y)))
}

#[cube(inline)]
fn c_sqrt(a: C2) -> C2 {
    let (x, y) = (a.extract(0usize), a.extract(1usize));
    let r = f32::hypot(x, y);
    // The principal root, as CUDA's helper computes it: the larger component
    // from the square root, the other from y / (2 * it), avoiding
    // cancellation.
    let re_big = f32::sqrt(0.5 * (r + x));
    let im_big = f32::sqrt(0.5 * (r - x));
    let im_big = select(y < 0.0, -im_big, im_big);
    let positive = x >= 0.0;
    let re = select(
        positive,
        re_big,
        select(im_big == 0.0, 0.0, y / (2.0 * im_big)),
    );
    let im = select(
        positive,
        select(re_big == 0.0, 0.0, y / (2.0 * re_big)),
        im_big,
    );
    c_make(re, im)
}

#[cube(inline)]
fn c_tanh(a: C2) -> C2 {
    let (x2, y2) = (2.0 * a.extract(0usize), 2.0 * a.extract(1usize));
    let d = f32::cosh(x2) + f32::cos(y2);
    c_make(f32::sinh(x2) / d, f32::sin(y2) / d)
}

#[cube(inline)]
fn c_pow(a: C2, b: C2) -> C2 {
    c_exp(c_mul(b, c_log(a)))
}
