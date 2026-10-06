use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    frontend::{clamp, max, min},
    prelude::{Cast, CubePartialOrd, CubePrimitive, CubeType, NativeExpand, Scope},
    unexpanded,
};

/// A value stored on the device as a native primitive, such as a unit or ID
/// brand over `f32` or `u32`.
///
/// Every `Repr` value must be a valid `Self`: storage loads and host reads
/// rebuild values with [`from_repr`](Self::from_repr) and never validate them.
/// A refinement such as a positive distance stores its unrestricted quantity
/// instead and is checked where it's constructed.
pub trait DeviceRepr: CubeType + Sized + 'static {
    type Repr: CubePrimitive;

    /// Present when `Self` has the layout of `Repr`, so host slices convert
    /// in place instead of being copied.
    const TRANSPARENT: Option<Transparent<Self>> = None;

    fn from_repr(value: Self::Repr) -> Self;
    fn into_repr(self) -> Self::Repr;
    #[doc(hidden)]
    fn __expand_from_repr(
        _: &Scope,
        value: <Self::Repr as CubeType>::ExpandType,
    ) -> Self::ExpandType {
        Self::expand_from_repr(value)
    }

    #[doc(hidden)]
    fn __expand_into_repr(
        _: &Scope,
        value: Self::ExpandType,
    ) -> <Self::Repr as CubeType>::ExpandType {
        Self::expand_into_repr(value)
    }

    #[doc(hidden)]
    fn expand_from_repr(value: <Self::Repr as CubeType>::ExpandType) -> Self::ExpandType;
    #[doc(hidden)]
    fn expand_into_repr(value: Self::ExpandType) -> <Self::Repr as CubeType>::ExpandType;
}

impl<T: CubePrimitive> DeviceRepr for T {
    type Repr = T;
    // SAFETY: a value is its own representation.
    const TRANSPARENT: Option<Transparent<Self>> = Some(unsafe { Transparent::new() });

    fn from_repr(value: T) -> Self {
        value
    }

    fn into_repr(self) -> T {
        self
    }

    fn expand_from_repr(value: Self::ExpandType) -> Self::ExpandType {
        value
    }

    fn expand_into_repr(value: Self::ExpandType) -> Self::ExpandType {
        value
    }
}

/// Proof that `Q` has the size, alignment and validity of `Q::Repr`.
pub struct Transparent<Q>(PhantomData<Q>);

impl<Q> Clone for Transparent<Q> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q> Copy for Transparent<Q> {}

impl<Q: DeviceRepr> Transparent<Q> {
    /// # Safety
    ///
    /// `Q` must be `Q::Repr` or `#[repr(transparent)]` over it, and every
    /// `Q::Repr` value must be a valid `Q`.
    pub const unsafe fn new() -> Self {
        Self(PhantomData)
    }

    pub fn reprs(self, values: &[Q]) -> &[Q::Repr] {
        // SAFETY: the layouts match, as `new` requires.
        unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
    }

    pub fn values(self, reprs: Vec<Q::Repr>) -> Vec<Q> {
        let mut reprs = core::mem::ManuallyDrop::new(reprs);
        // SAFETY: the layouts match and every representation is a valid `Q`,
        // as `new` requires, so the allocation is reused as is.
        unsafe { Vec::from_raw_parts(reprs.as_mut_ptr().cast(), reprs.len(), reprs.capacity()) }
    }
}

/// Same-brand ordering of a [`DeviceRepr`] value, computed on its native
/// representation. Derived with `#[device_repr(ord)]`.
pub trait Ordered: DeviceRepr<Repr: CubePartialOrd> + PartialOrd
where
    Self: CubeType<ExpandType: OrderedExpand>,
{
    fn min(self, _other: Self) -> Self {
        unexpanded!()
    }

    fn max(self, _other: Self) -> Self {
        unexpanded!()
    }

    fn clamp(self, _min: Self, _max: Self) -> Self {
        unexpanded!()
    }
}

#[doc(hidden)]
pub trait OrderedExpand: Sized {
    type Value: DeviceRepr<ExpandType = Self, Repr: CubePartialOrd>;

    fn __expand_min_method(self, scope: &Scope, other: Self) -> Self {
        let [lhs, rhs] = [self, other].map(Self::Value::expand_into_repr);
        Self::Value::expand_from_repr(min::expand(scope, lhs, rhs))
    }

    fn __expand_max_method(self, scope: &Scope, other: Self) -> Self {
        let [lhs, rhs] = [self, other].map(Self::Value::expand_into_repr);
        Self::Value::expand_from_repr(max::expand(scope, lhs, rhs))
    }

    fn __expand_clamp_method(self, scope: &Scope, min: Self, max: Self) -> Self {
        let [value, min, max] = [self, min, max].map(Self::Value::expand_into_repr);
        Self::Value::expand_from_repr(clamp::expand(scope, value, min, max))
    }
}

/// Addresses the elements of a [`Storage`](crate::prelude::Storage), so a
/// table can accept only its own ID type. Derived for `u32` and `usize` brands
/// with `#[device_repr(key)]`.
pub trait StorageKey: CubeType + 'static {
    #[doc(hidden)]
    fn __expand_position(scope: &Scope, key: Self::ExpandType) -> NativeExpand<usize>;
}

impl StorageKey for usize {
    fn __expand_position(_: &Scope, key: NativeExpand<usize>) -> NativeExpand<usize> {
        key
    }
}

impl StorageKey for u32 {
    fn __expand_position(scope: &Scope, key: NativeExpand<u32>) -> NativeExpand<usize> {
        usize::__expand_cast_from(scope, key)
    }
}

/// The same brand over another native representation, such as vectors of its
/// scalar. Derived when the wrapped field's type is a type parameter.
pub trait WithRepr<R: CubePrimitive>: DeviceRepr {
    type Output: DeviceRepr<Repr = R>;
}
