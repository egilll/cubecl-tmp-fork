//! Device functions: a `#[cube]` function traced once per specialization and
//! called from every call site.
//!
//! Tracing a call into its caller runs the callee's expansion again there, so
//! a helper called at 64 sites would be traced 64 times and its body would
//! appear 64 times in the kernel. Instead, a call traces the body into a
//! private function of the kernel module the first time, and emits a call
//! every time. Whether those calls survive to the target is the inliner's
//! decision (`cubecl_opt::passes::inline`).
//!
//! Only runtime arguments become parameters. A constant argument (a literal,
//! or a value folded while tracing) is part of the specialization instead, so
//! the body is traced with it exactly as an inlined call would be.
//!
//! A call is traced inline instead, as `#[cube(inline)]` asks, whenever it
//! can't be a call: an argument or the result isn't a [`CallArg`] that can be
//! passed, a comptime argument can't be hashed into the specialization, or the
//! traced body doesn't stand on its own (see [`InlineReason`]).

use alloc::{string::String, vec::Vec};
use core::hash::{Hash, Hasher};

use cubecl_ir::{
    AddressSpace, ExpandValue, FuncOpExt, OpInserter, Scope,
    convert::closure_captures,
    device_fn::{CallArgKey, CallKey, DeviceFn, InlineReason, hash_unordered},
    dialect::{
        branch::ReturnOp,
        call::CallOp,
        memory::{LoadOp, StoreOp},
    },
    pliron::{
        builtin::{ops::FuncOp, types::FunctionType},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        value::Value,
    },
};

use crate::prelude::NativeExpand;
use cubecl_ir::{
    SliceMetadata,
    dialect::vector::CompositeConstructOp,
    interfaces::TypedExt,
    types::{PointerType, aggregate::SliceType},
};

pub use cubecl_ir::device_fn::call_hasher;

/// One runtime argument of a call, once read in the caller.
#[derive(Debug, Clone, Copy)]
pub enum CallSlot {
    /// Passed to the function as a parameter.
    Param(Value),
    /// Known while tracing: part of the specialization, not a parameter.
    Constant(ExpandValue),
}

/// A value a device function can take or return. Every expand type implements
/// it; the defaults say it can't, and a call involving one is traced inline.
///
/// A type that can be passed implements [`call_slots`](Self::call_slots) and
/// [`call_rebuild`](Self::call_rebuild); one that can be returned implements
/// [`call_returns`](Self::call_returns) and
/// [`call_from_results`](Self::call_from_results). `#[derive(CubeType)]`
/// passes a struct as its runtime fields.
pub trait CallArg: Sized {
    /// Read this value in the caller as the slots a call passes, and say
    /// whether it can be passed at all. Slots pushed before a `false` are
    /// discarded with the call.
    fn call_slots(&self, _scope: &Scope, _slots: &mut Vec<CallSlot>) -> bool {
        false
    }

    /// Hash what of this value is known while tracing and isn't a slot, such
    /// as a struct's comptime fields, and say whether it could: a value whose
    /// type isn't known to be [`Hash`] can't be part of a specialization.
    fn call_key(&self, _hasher: &mut dyn Hasher) -> bool {
        true
    }

    /// This value inside the function: parameters for the slots that are
    /// parameters, taken in order from `params`, and the rest as it is. Only
    /// called after [`call_slots`](Self::call_slots) said it can be passed.
    fn call_rebuild(&self, _scope: &Scope, _params: &mut dyn Iterator<Item = Value>) -> Self {
        unreachable!("a value that can't be passed is never rebuilt")
    }

    /// The values the function returns for this result, read inside it, or
    /// `None` when a function can't return it.
    fn call_returns(&self, _scope: &Scope) -> Option<Vec<Value>> {
        None
    }

    /// The result at the call site, from the call's results. Only called for
    /// a type whose [`call_returns`](Self::call_returns) answered.
    fn call_from_results(_results: &mut dyn Iterator<Item = Value>) -> Self {
        unreachable!("a value that can't be returned is never rebuilt from results")
    }
}

impl CallArg for () {
    fn call_returns(&self, _scope: &Scope) -> Option<Vec<Value>> {
        Some(Vec::new())
    }

    fn call_from_results(_results: &mut dyn Iterator<Item = Value>) -> Self {}
}

/// A reference is passed by the macro as the value it refers to, which the
/// body borrows again, so the reference itself is never an argument.
impl<T: ?Sized> CallArg for &T {}
impl<T: ?Sized> CallArg for &mut T {}

/// A scalar or a vector passes by value, as Rust passes a `Copy` value. A
/// slice passes the buffer it points into, its offset and its length: the
/// buffer pointer's type names the buffer, so a function taking a slice is
/// specialized per buffer, and a mutable slice writes in place, which is what
/// the caller sees either way. Other native values (arrays, shared memory,
/// barriers, atomics) are places a function would have to share with its
/// caller, so a call taking one is traced inline.
impl<T: ?Sized> CallArg for NativeExpand<T> {
    fn call_slots(&self, scope: &Scope, slots: &mut Vec<CallSlot>) -> bool {
        let value = match self.expand {
            ExpandValue::Constant { .. } => {
                slots.push(CallSlot::Constant(self.expand));
                return true;
            }
            ExpandValue::Value(value) => value,
        };
        match kind(scope, value) {
            Kind::ByValue => {
                // Read here, in the caller: the function takes a copy.
                slots.push(CallSlot::Param(self.expand.read_value(scope)));
                true
            }
            Kind::Slice => {
                let slice = self.expand.read_value(scope);
                for field in [
                    SliceMetadata::LIST,
                    SliceMetadata::OFFSET,
                    SliceMetadata::LENGTH,
                ] {
                    slots.push(CallSlot::Param(scope.extract_field(slice, field)));
                }
                true
            }
            Kind::Place => false,
        }
    }

    fn call_rebuild(&self, scope: &Scope, params: &mut dyn Iterator<Item = Value>) -> Self {
        let mut next = || params.next().expect("one parameter per slot");
        let value = match self.expand {
            ExpandValue::Constant { .. } => return self.expand.into(),
            ExpandValue::Value(value) => value,
        };
        match kind(scope, value) {
            Kind::ByValue => next().into(),
            Kind::Slice => {
                let (list, offset, length) = (next(), next(), next());
                let ty = SliceType::get(scope.ctx(), list.get_type(scope.ctx())).to_handle();
                let op = CompositeConstructOp::new(
                    scope.ctx_mut(),
                    ty,
                    alloc::vec![list, offset, length],
                );
                scope.register_with_result(&op).into()
            }
            Kind::Place => unreachable!("a place is never passed"),
        }
    }

    fn call_returns(&self, scope: &Scope) -> Option<Vec<Value>> {
        let value = match self.expand {
            ExpandValue::Constant { .. } => return Some(alloc::vec![self.expand.value(scope)]),
            ExpandValue::Value(value) => value,
        };
        match kind(scope, value) {
            Kind::ByValue => Some(alloc::vec![self.expand.read_value(scope)]),
            Kind::Slice | Kind::Place => None,
        }
    }

    fn call_from_results(results: &mut dyn Iterator<Item = Value>) -> Self {
        results
            .next()
            .expect("one result per returned value")
            .into()
    }
}

/// How a native value passes to a device function.
enum Kind {
    /// A scalar or a vector, copied.
    ByValue,
    /// A slice, as its parts.
    Slice,
    /// Anything else: not passed.
    Place,
}

/// What `value` holds, or points to.
fn kind(scope: &Scope, value: Value) -> Kind {
    let ctx = scope.ctx();
    let ty = value.get_type(ctx);
    let ty = match ty.deref(ctx).downcast_ref::<PointerType>() {
        Some(pointer) => pointer.inner,
        None => ty,
    };
    if ty.deref(ctx).is::<SliceType>() {
        Kind::Slice
    } else if ty.is_vector(ctx)
        || ty.is_int(ctx)
        || ty.is_index(ctx)
        || ty.is_float(ctx)
        || ty.is_bool(ctx)
    {
        Kind::ByValue
    } else {
        Kind::Place
    }
}

/// Hashes a comptime value into a call's specialization when its type is
/// known to be [`Hash`], and says whether it could.
///
/// Used through autoref dispatch, as `(&HashProbe(&value)).hash_comptime(h)`
/// with [`HashComptime`] and [`NoHashComptime`] in scope: for a type known to
/// be `Hash` the first applies, otherwise (an unbounded generic, say) the
/// second, and the call is traced inline.
pub struct HashProbe<'a, T: ?Sized>(pub &'a T);

/// See [`HashProbe`].
pub trait HashComptime {
    fn hash_comptime(&self, hasher: &mut dyn Hasher) -> bool;
}

impl<T: Hash + ?Sized> HashComptime for HashProbe<'_, T> {
    fn hash_comptime(&self, mut hasher: &mut dyn Hasher) -> bool {
        self.0.hash(&mut hasher);
        true
    }
}

/// See [`HashProbe`].
pub trait NoHashComptime {
    fn hash_comptime(&self, _hasher: &mut dyn Hasher) -> bool {
        false
    }
}

impl<T: ?Sized> NoHashComptime for &HashProbe<'_, T> {}

/// Hash one const generic into `hasher`, for [`device_call`]'s `comptime`.
pub fn hash_const<T: Hash + ?Sized>(hasher: &mut impl Hasher, value: &T) {
    value.hash(hasher);
}

/// The arguments of a call, read in the caller.
pub struct CallArgs {
    /// Each runtime argument's slots, in order.
    pub slots: Vec<CallSlot>,
    /// A hash of what the arguments carry besides their slots.
    pub key: u64,
}

/// Call the device function `function` with `args`, tracing it the first time
/// this specialization is called in the kernel; `None` when the call has to be
/// traced inline instead.
///
/// `generics` names the function's generic types and `comptime` hashes its
/// comptime arguments: with the runtime arguments' types, they decide which
/// calls share a function. `args` is `None` when an argument can't be passed
/// or a comptime argument can't be hashed. `trace` traces the body into the
/// function, rebuilding the arguments from its parameters; it runs at most
/// once. When the traced body turns out not to stand on its own (it used a
/// value of its caller, terminates the kernel, or returns something a
/// function can't), the function is discarded and the answer is `None`, so
/// the caller traces the body inline, as `#[cube(inline)]` would.
pub fn device_call<R: CallArg>(
    scope: &Scope,
    function: &'static str,
    generics: &[&'static str],
    comptime: u64,
    args: Option<CallArgs>,
    trace: impl FnOnce(&Scope, &mut dyn Iterator<Item = Value>) -> R,
) -> Option<R> {
    if !scope.state().traces_device_fns {
        return None;
    }
    let CallArgs {
        slots,
        key: arg_key,
    } = args?;
    let params: Vec<Value> = slots
        .iter()
        .filter_map(|slot| match slot {
            CallSlot::Param(value) => Some(*value),
            CallSlot::Constant(_) => None,
        })
        .collect();

    let key = {
        let ctx = scope.ctx();
        let state = scope.state();
        CallKey {
            function,
            generics: generics.to_vec(),
            comptime,
            args: slots
                .iter()
                .map(|slot| match slot {
                    CallSlot::Param(value) => CallArgKey::Value(value.get_type(ctx)),
                    CallSlot::Constant(ExpandValue::Constant { value, ty }) => {
                        CallArgKey::Constant((*value).into(), *ty)
                    }
                    CallSlot::Constant(ExpandValue::Value(_)) => {
                        unreachable!("a constant slot holds a constant")
                    }
                })
                .collect(),
            modes: state.modes,
            arg_comptime: arg_key,
            registrations: hash_unordered(state.typemap.iter())
                ^ hash_unordered(state.sizemap.iter()).rotate_left(1),
        }
    };

    match scope.state().device_fns.get(&key).cloned() {
        Some(DeviceFn::Function { symbol, ty, outs }) => {
            return Some(emit_call(scope, symbol, ty, params, &outs));
        }
        Some(DeviceFn::Inline(_)) => return None,
        None => {}
    }

    let param_types: Vec<TypeHandle> = params.iter().map(|it| it.get_type(scope.ctx())).collect();
    let name = scope.func_ident(Some(short_name(function).as_str()));
    let placeholder_ty = FunctionType::get(scope.ctx(), param_types.clone(), Vec::new());
    let func = FuncOp::new(scope.ctx_mut(), name.clone(), placeholder_ty);
    let entry = func.get_entry_block(scope.ctx());
    let block_args: Vec<Value> = entry.deref(scope.ctx()).arguments().collect();

    let child = scope.func_child(OpInserter::new_at_block_end(entry));
    let result = trace(&child, &mut block_args.into_iter());
    let returned = result.call_returns(&child);
    let terminates = child.expand_state().may_return;
    // A function returns one value. The rest of a struct's fields leave
    // through pointers to locals of the caller, appended as parameters: they
    // are fresh and only written, so nothing aliases, and once a call is
    // inlined the locals are promoted away.
    let mut outs = Vec::new();
    if let Some(returned) = &returned {
        for value in returned.iter().skip(1) {
            let value_ty = value.get_type(scope.ctx());
            let ptr_ty = PointerType::get(scope.ctx(), value_ty, AddressSpace::Local);
            let idx = func.push_argument(scope.ctx(), ptr_ty.into());
            let ptr = func
                .get_entry_block(scope.ctx())
                .deref(scope.ctx())
                .get_argument(idx);
            child.register(&StoreOp::new(scope.ctx_mut(), ptr, *value));
            outs.push(value_ty);
        }
        let ret = match returned.first() {
            None => ReturnOp::new(scope.ctx_mut()),
            Some(value) => ReturnOp::new_with_value(scope.ctx_mut(), *value),
        };
        child.register(&ret);
    }
    drop(child);

    let reason = if returned.is_none() {
        Some(InlineReason::Returns)
    } else if terminates {
        Some(InlineReason::Terminates)
    } else if !closure_captures(scope.ctx(), &func).is_empty() {
        Some(InlineReason::Captures)
    } else {
        None
    };
    if let Some(reason) = reason {
        Operation::erase(func.get_operation(), scope.ctx_mut());
        scope
            .state_mut()
            .device_fns
            .insert(key, DeviceFn::Inline(reason));
        return None;
    }

    let returned = returned.expect("checked above");
    let result_types = returned
        .first()
        .map(|it| it.get_type(scope.ctx()))
        .into_iter()
        .collect();
    let all_params = param_types
        .into_iter()
        .chain(
            outs.iter()
                .map(|ty| PointerType::get(scope.ctx(), *ty, AddressSpace::Local).into()),
        )
        .collect();
    let ty: TypeHandle = FunctionType::get(scope.ctx(), all_params, result_types).into();
    func.set_attr_builtin_func_type(scope.ctx(), ty.into());

    scope.register_func(func);
    scope.state_mut().device_fns.insert(
        key,
        DeviceFn::Function {
            symbol: name.clone(),
            ty,
            outs: outs.clone(),
        },
    );
    Some(emit_call(scope, name, ty, params, &outs))
}

fn emit_call<R: CallArg>(
    scope: &Scope,
    symbol: Identifier,
    ty: TypeHandle,
    mut params: Vec<Value>,
    outs: &[TypeHandle],
) -> R {
    let locals: Vec<Value> = outs
        .iter()
        .map(|ty| scope.create_local_mut(*ty, None))
        .collect();
    params.extend(locals.iter().copied());
    let call = CallOp::new(scope.ctx_mut(), symbol, ty, params);
    scope.register(&call);
    let mut results: Vec<Value> = call.get_operation().deref(scope.ctx()).results().collect();
    for local in locals {
        let load = LoadOp::new(scope.ctx_mut(), local);
        results.push(scope.register_with_result(&load));
    }
    R::call_from_results(&mut results.into_iter())
}

/// The last path segment of `function`, cleaned up for a symbol name.
fn short_name(function: &str) -> String {
    let name = function
        .rsplit("::")
        .map(str::trim)
        .find(|it| *it != "__kernel" && !it.is_empty());
    let name = name.unwrap_or("device_fn");
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// What generated `CallArg` impls name, without depending on the caller's
/// imports.
#[doc(hidden)]
pub mod __private {
    pub use alloc::vec::Vec;
    pub use cubecl_ir::{Scope, pliron::value::Value};
}
