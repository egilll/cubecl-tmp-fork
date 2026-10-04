use core::fmt::Debug;

use cubecl_core::ir::{
    ContextExt, FastMath,
    attributes::{ATTR_FAST_MATH, FastMathAttr},
    dialect::OperationPtrExt,
    interfaces::TypedExt,
};
use pliron::{context::Context, r#type::Typed};
use pliron::{context::Ptr, operation::Operation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cuda,
    Hip,
    Metal,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Shared;
#[derive(Debug, Clone, Copy, Default)]
pub struct Cuda;
#[derive(Debug, Clone, Copy, Default)]
pub struct Hip;
#[derive(Debug, Clone, Copy, Default)]
pub struct Metal;

impl Target {
    pub fn ty_prefix(&self, ctx: &Context, ty: impl Typed) -> &'static str {
        if ty.is_half(ctx) {
            self.half_prefix()
        } else if ty.is_half2(ctx) {
            self.half2_prefix()
        } else {
            ""
        }
    }

    /// The namespace that pins a `float` math function to a variant. On Metal,
    /// a plain `exp` is the fast or the precise one depending on the compile
    /// options, and wgpu's passthrough and the native runtime set those
    /// differently, so the source names the variant itself: `fast::` for an
    /// operation created under `fast_math` with reduced precision allowed,
    /// `precise::` otherwise. Only `float` has both.
    pub fn math_prefix(&self, ctx: &Context, op: Ptr<Operation>, ty: impl Typed) -> &'static str {
        match self {
            Target::Metal if ty.scalar_ty(ctx).is_float32(ctx) => {
                let fast = op
                    .get_attr::<FastMathAttr>(ctx, &ATTR_FAST_MATH)
                    .is_some_and(|attr| attr.allows(FastMath::ReducedPrecision));
                match fast {
                    true => "fast::",
                    false => "precise::",
                }
            }
            _ => "",
        }
    }

    pub fn half_prefix(&self) -> &'static str {
        match self {
            Target::Cuda | Target::Hip => "h",
            Target::Metal => "",
        }
    }

    pub fn half2_prefix(&self) -> &'static str {
        match self {
            Target::Cuda | Target::Hip => "h2",
            Target::Metal => "",
        }
    }
}

pub trait CppTarget: Default + Clone + Copy + Debug + Send + Sync + 'static {
    fn target() -> Target;
}

impl CppTarget for Cuda {
    fn target() -> Target {
        Target::Cuda
    }
}
impl CppTarget for Hip {
    fn target() -> Target {
        Target::Hip
    }
}
impl CppTarget for Metal {
    fn target() -> Target {
        Target::Metal
    }
}

impl CtxTarget for Context {}
pub trait CtxTarget: ContextExt {
    fn target(&self) -> Target {
        *self.aux_ty::<Target>()
    }
    fn set_target(&mut self, value: Target) {
        self.set_aux_ty(value);
    }
}

macro_rules! dispatch_target {
    ($ctx: expr, $expr: expr) => {{
        use $crate::target::CtxTarget;
        match $ctx.target() {
            $crate::target::Target::Cuda => {
                type Target = $crate::target::Cuda;
                $expr
            }
            $crate::target::Target::Hip => {
                type Target = $crate::target::Hip;
                $expr
            }
            $crate::target::Target::Metal => {
                type Target = $crate::target::Metal;
                $expr
            }
        }
    }};
}
pub(crate) use dispatch_target;

use crate::shared::ty::TypedExtCPP;
