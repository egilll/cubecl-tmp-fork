//! Plane reductions and scans.

use crate::prelude::*;
use cubecl_core::{
    ir::dialect::plane,
    prelude::{
        polyfills::plane::{
            OpAdd, OpMax, OpMin, OpMul, plane_reduce, plane_reduce_exclusive,
            plane_reduce_inclusive, plane_single, plane_single_exclusive,
        },
        *,
    },
};

define_scalar!(T);
define_size!(S);

macro_rules! lower_reduction {
    ($ty:ty, $reduce:ident, $single:ident, $op:ty $(, $args:expr)*) => {
        #[op_interface_impl]
        impl LowerOp for $ty {
            fn should_lower(&self, _ctx: &Context) -> bool {
                true
            }

            fn lower(&self, scope: &Scope) -> Vec<Value> {
                let input = self.input(scope.ctx());
                scope.register_value_type::<T, S>(input);
                // A CPU plane is one unit: the scan has nothing to exchange.
                if scope.ctx().target() == LlvmTarget::Cpu {
                    return vec![
                        $single::expand::<T, S>(scope, input.into() $(, $args)*).read_value(scope),
                    ];
                }
                vec![
                    $reduce::expand::<T, S, $op>(scope, input.into() $(, $args)*)
                        .read_value(scope),
                ]
            }
        }
    };
}

lower_reduction!(plane::ISumOp, plane_reduce, plane_single, OpAdd);
lower_reduction!(plane::FSumOp, plane_reduce, plane_single, OpAdd);
lower_reduction!(plane::IProdOp, plane_reduce, plane_single, OpMul);
lower_reduction!(plane::FProdOp, plane_reduce, plane_single, OpMul);
lower_reduction!(plane::SMinOp, plane_reduce, plane_single, OpMin);
lower_reduction!(plane::UMinOp, plane_reduce, plane_single, OpMin);
lower_reduction!(plane::FMinOp, plane_reduce, plane_single, OpMin);
lower_reduction!(plane::SMaxOp, plane_reduce, plane_single, OpMax);
lower_reduction!(plane::UMaxOp, plane_reduce, plane_single, OpMax);
lower_reduction!(plane::FMaxOp, plane_reduce, plane_single, OpMax);

lower_reduction!(
    plane::InclusiveISumOp,
    plane_reduce_inclusive,
    plane_single,
    OpAdd
);
lower_reduction!(
    plane::InclusiveFSumOp,
    plane_reduce_inclusive,
    plane_single,
    OpAdd
);
lower_reduction!(
    plane::InclusiveIProdOp,
    plane_reduce_inclusive,
    plane_single,
    OpMul
);
lower_reduction!(
    plane::InclusiveFProdOp,
    plane_reduce_inclusive,
    plane_single,
    OpMul
);

lower_reduction!(
    plane::ExclusiveISumOp,
    plane_reduce_exclusive,
    plane_single_exclusive,
    OpAdd,
    0
);
lower_reduction!(
    plane::ExclusiveFSumOp,
    plane_reduce_exclusive,
    plane_single_exclusive,
    OpAdd,
    0
);
lower_reduction!(
    plane::ExclusiveIProdOp,
    plane_reduce_exclusive,
    plane_single_exclusive,
    OpMul,
    1
);
lower_reduction!(
    plane::ExclusiveFProdOp,
    plane_reduce_exclusive,
    plane_single_exclusive,
    OpMul,
    1
);
