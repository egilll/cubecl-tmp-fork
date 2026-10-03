//! Device functions traced once per specialization and called from every
//! call site, for `#[cube(outline)]`.
//!
//! A plain `#[cube]` call runs the callee's expansion again in its caller, so
//! a helper called at 64 sites is traced 64 times and its body appears 64
//! times in the kernel. An outlined call traces the body into a private
//! function of the kernel module the first time, and emits a call every time.
//! Whether those calls survive to the target is the inliner's decision.
//!
//! Only runtime arguments become parameters. A constant argument (a literal,
//! or a value folded while tracing) is part of the specialization instead, so
//! the body is traced with it exactly as an inlined call would be.

use alloc::{string::String, vec::Vec};
use core::hash::{Hash, Hasher};

use cubecl_ir::{
    ExpandValue, OpInserter, Scope,
    convert::closure_captures,
    dialect::{branch::ReturnOp, call::CallOp},
    outline::{OutlineArgKey, OutlineFallback, OutlineKey, Outlined, hash_unordered},
    pliron::{
        builtin::{ops::FuncOp, types::FunctionType},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
};

use crate::prelude::{CubePrimitive, NativeExpand};

/// One runtime argument of an outlined call, once read in the caller.
#[derive(Debug, Clone, Copy)]
pub enum OutlineSlot {
    /// Passed to the function as a parameter.
    Param(Value),
    /// Known while tracing: part of the specialization, not a parameter.
    Constant(ExpandValue),
}

/// An argument an outlined function can take.
#[diagnostic::on_unimplemented(
    message = "`#[cube(outline)]` functions take scalars and vectors by value, and `{Self}` is not one",
    label = "not supported by an outlined function yet",
    note = "remove `outline` from the function's `#[cube]` attribute to inline it at each call"
)]
pub trait OutlineArg: Sized {
    /// Read this argument in the caller, as the slots the call passes.
    fn outline_slots(&self, scope: &Scope, slots: &mut Vec<OutlineSlot>);

    /// This argument inside the function: parameters for the slots that are
    /// parameters, taken in order from `params`, and the constants as they
    /// are.
    fn outline_rebuild(&self, params: &mut dyn Iterator<Item = Value>) -> Self;
}

impl<T: CubePrimitive> OutlineArg for NativeExpand<T> {
    fn outline_slots(&self, scope: &Scope, slots: &mut Vec<OutlineSlot>) {
        slots.push(match self.expand {
            ExpandValue::Constant { .. } => OutlineSlot::Constant(self.expand),
            // Read here, in the caller: the function takes a copy, as Rust
            // passes a `Copy` value.
            ExpandValue::Value(_) => OutlineSlot::Param(self.expand.read_value(scope)),
        });
    }

    fn outline_rebuild(&self, params: &mut dyn Iterator<Item = Value>) -> Self {
        match self.expand {
            ExpandValue::Constant { .. } => self.expand.into(),
            ExpandValue::Value(_) => params
                .next()
                .expect("one parameter per runtime argument")
                .into(),
        }
    }
}

/// The runtime arguments of an outlined call, as a tuple.
pub trait OutlineArgs: Sized {
    fn outline_slots(&self, scope: &Scope, slots: &mut Vec<OutlineSlot>);
    fn outline_rebuild(&self, params: &mut dyn Iterator<Item = Value>) -> Self;
}

macro_rules! outline_args_tuple {
    ($($name:ident),*) => {
        impl<$($name: OutlineArg),*> OutlineArgs for ($($name,)*) {
            #[allow(unused_variables, non_snake_case)]
            fn outline_slots(&self, scope: &Scope, slots: &mut Vec<OutlineSlot>) {
                let ($($name,)*) = self;
                $($name.outline_slots(scope, slots);)*
            }

            #[allow(unused_variables, non_snake_case, clippy::unused_unit)]
            fn outline_rebuild(&self, params: &mut dyn Iterator<Item = Value>) -> Self {
                let ($($name,)*) = self;
                ($($name.outline_rebuild(params),)*)
            }
        }
    };
}

outline_args_tuple!();
outline_args_tuple!(A);
outline_args_tuple!(A, B);
outline_args_tuple!(A, B, C);
outline_args_tuple!(A, B, C, D);
outline_args_tuple!(A, B, C, D, E);
outline_args_tuple!(A, B, C, D, E, F);
outline_args_tuple!(A, B, C, D, E, F, G);
outline_args_tuple!(A, B, C, D, E, F, G, H);
outline_args_tuple!(A, B, C, D, E, F, G, H, I);
outline_args_tuple!(A, B, C, D, E, F, G, H, I, J);
outline_args_tuple!(A, B, C, D, E, F, G, H, I, J, K);
outline_args_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);

/// What an outlined function can return.
#[diagnostic::on_unimplemented(
    message = "`#[cube(outline)]` functions return a scalar, a vector or nothing, and `{Self}` is not one",
    label = "not supported by an outlined function yet",
    note = "remove `outline` from the function's `#[cube]` attribute to inline it at each call"
)]
pub trait OutlineResult: Sized {
    /// The values the function returns, read inside it.
    fn outline_returns(&self, scope: &Scope) -> Vec<Value>;
    /// The result at the call site, from the call's results.
    fn outline_from_results(results: &mut dyn Iterator<Item = Value>) -> Self;
}

impl OutlineResult for () {
    fn outline_returns(&self, _scope: &Scope) -> Vec<Value> {
        Vec::new()
    }

    fn outline_from_results(_results: &mut dyn Iterator<Item = Value>) -> Self {}
}

impl<T: CubePrimitive> OutlineResult for NativeExpand<T> {
    fn outline_returns(&self, scope: &Scope) -> Vec<Value> {
        alloc::vec![self.expand.read_value(scope)]
    }

    fn outline_from_results(results: &mut dyn Iterator<Item = Value>) -> Self {
        results
            .next()
            .expect("one result per returned value")
            .into()
    }
}

/// Call the device function `function` with `args`, tracing `body` into it
/// the first time this specialization is called in the kernel.
///
/// `generics` names the function's generic types and `comptime` hashes its
/// comptime arguments: with the runtime arguments' types, they decide which
/// calls share a function. `body` is the function's expansion. It may run
/// twice: when the traced body turns out not to stand on its own (it used a
/// value of its caller, or terminates the kernel), the call falls back to
/// tracing `body` inline, as a plain `#[cube]` call does.
pub fn outline_call<A: OutlineArgs, R: OutlineResult>(
    scope: &Scope,
    function: &'static str,
    generics: &[&'static str],
    comptime: u64,
    args: A,
    body: impl Fn(&Scope, A) -> R,
) -> R {
    let mut slots = Vec::new();
    args.outline_slots(scope, &mut slots);
    let params: Vec<Value> = slots
        .iter()
        .filter_map(|slot| match slot {
            OutlineSlot::Param(value) => Some(*value),
            OutlineSlot::Constant(_) => None,
        })
        .collect();

    let key = {
        let ctx = scope.ctx();
        let state = scope.state();
        OutlineKey {
            function,
            generics: generics.to_vec(),
            comptime,
            args: slots
                .iter()
                .map(|slot| match slot {
                    OutlineSlot::Param(value) => OutlineArgKey::Value(value.get_type(ctx)),
                    OutlineSlot::Constant(ExpandValue::Constant { value, ty }) => {
                        OutlineArgKey::Constant((*value).into(), *ty)
                    }
                    OutlineSlot::Constant(ExpandValue::Value(_)) => {
                        unreachable!("a constant slot holds a constant")
                    }
                })
                .collect(),
            modes: state.modes,
            registrations: hash_unordered(state.typemap.iter())
                ^ hash_unordered(state.sizemap.iter()).rotate_left(1),
        }
    };

    match scope.state().outlined.get(&key).cloned() {
        Some(Outlined::Function { symbol, ty }) => return emit_call(scope, symbol, ty, params),
        Some(Outlined::Inline(_)) => return body(scope, args),
        None => {}
    }

    let param_types: Vec<TypeHandle> = params.iter().map(|it| it.get_type(scope.ctx())).collect();
    let name = scope.func_ident(Some(short_name(function).as_str()));
    let placeholder_ty = FunctionType::get(scope.ctx(), param_types.clone(), Vec::new());
    let func = FuncOp::new(scope.ctx_mut(), name.clone(), placeholder_ty);
    let entry = func.get_entry_block(scope.ctx());
    let block_args: Vec<Value> = entry.deref(scope.ctx()).arguments().collect();

    let child = scope.func_child(OpInserter::new_at_block_end(entry));
    let result = body(&child, args.outline_rebuild(&mut block_args.into_iter()));
    let returned = result.outline_returns(&child);
    let terminates = child.expand_state().may_return;
    let ret = match returned.as_slice() {
        [] => ReturnOp::new(scope.ctx_mut()),
        [value] => ReturnOp::new_with_value(scope.ctx_mut(), *value),
        _ => unreachable!("an outlined function returns at most one value"),
    };
    child.register(&ret);
    drop(child);

    let result_types = returned.iter().map(|it| it.get_type(scope.ctx())).collect();
    let ty: TypeHandle = FunctionType::get(scope.ctx(), param_types, result_types).into();
    func.set_attr_builtin_func_type(scope.ctx(), ty.into());

    let fallback = if terminates {
        Some(OutlineFallback::Terminates)
    } else if !closure_captures(scope.ctx(), &func).is_empty() {
        Some(OutlineFallback::Captures)
    } else {
        None
    };
    if let Some(reason) = fallback {
        Operation::erase(func.get_operation(), scope.ctx_mut());
        scope
            .state_mut()
            .outlined
            .insert(key, Outlined::Inline(reason));
        return body(scope, args);
    }

    scope.register_func(func);
    scope.state_mut().outlined.insert(
        key,
        Outlined::Function {
            symbol: name.clone(),
            ty,
        },
    );
    emit_call(scope, name, ty, params)
}

fn emit_call<R: OutlineResult>(
    scope: &Scope,
    symbol: Identifier,
    ty: TypeHandle,
    params: Vec<Value>,
) -> R {
    let call = CallOp::new(scope.ctx_mut(), symbol, ty, params);
    scope.register(&call);
    let results: Vec<Value> = call.get_operation().deref(scope.ctx()).results().collect();
    R::outline_from_results(&mut results.into_iter())
}

/// The last path segment of `function`, cleaned up for a symbol name.
fn short_name(function: &str) -> String {
    let name = function
        .rsplit("::")
        .find(|it| *it != "__kernel" && !it.is_empty());
    let name = name.unwrap_or("outlined");
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// Hash one comptime argument into `hasher`, for [`outline_call`]'s
/// `comptime`.
pub fn outline_hash_comptime<T: Hash + ?Sized>(hasher: &mut impl Hasher, value: &T) {
    value.hash(hasher);
}

pub use cubecl_ir::outline::outline_hasher;
