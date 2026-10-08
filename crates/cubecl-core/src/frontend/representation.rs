use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    self as cubecl,
    frontend::{clamp, max, min},
    prelude::*,
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
    /// How values sit in a buffer: [`Native`], or [`Packed`] lanes.
    type Layout: StorageLayout<Self::Repr>;

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
    type Layout = Native;
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

/// The element a buffer of `Q` binds.
pub type StorageElement<Q> =
    <<Q as DeviceRepr>::Layout as StorageLayout<<Q as DeviceRepr>::Repr>>::Element;

/// Elements per value in a buffer of `Q`.
pub type StorageWidth<Q> =
    <<Q as DeviceRepr>::Layout as StorageLayout<<Q as DeviceRepr>::Repr>>::Width;

/// How a representation `R` is laid out in a buffer of [`Self::Element`]s.
#[cube]
#[diagnostic::on_unimplemented(
    message = "`{Self}` can't lay out `{R}` in storage",
    note = "`Packed` lays out vectors only"
)]
pub trait StorageLayout<R: CubePrimitive>: Send + Sync + 'static {
    type Element: CubePrimitive;
    /// Elements per value.
    type Width: Size;

    fn load_from(elements: &[Self::Element], position: usize) -> R;
    fn store_into(elements: &mut [Self::Element], position: usize, value: R);
}

/// One element per value: scalars, and vectors at their aligned size.
pub struct Native;

#[cube]
impl<R: CubePrimitive> StorageLayout<R> for Native {
    type Element = R;
    type Width = Const<1>;

    fn load_from(elements: &[R], position: usize) -> R {
        elements[position]
    }

    fn store_into(elements: &mut [R], position: usize, value: R) {
        elements[position] = value;
    }
}

/// Vectors as consecutive scalars, without the padding three-lane vectors
/// have in storage.
pub struct Packed;

#[cube]
impl<E: Scalar, N: Size> StorageLayout<Vector<E, N>> for Packed {
    type Element = E;
    type Width = N;

    fn load_from(elements: &[E], position: usize) -> Vector<E, N> {
        let lanes = N::value();
        let at = position * lanes;
        let mut value = Vector::<E, N>::new(elements[at]);
        #[unroll]
        for lane in 1..lanes {
            value.insert(lane, elements[at + lane]);
        }
        value
    }

    fn store_into(elements: &mut [E], position: usize, value: Vector<E, N>) {
        let lanes = N::value();
        let at = position * lanes;
        #[unroll]
        for lane in 0..lanes {
            elements[at + lane] = value.extract(lane);
        }
    }
}

/// A struct whose fields are the lanes of a packed vector, such as a complex
/// number's real and imaginary parts. It is stored as consecutive scalars;
/// kernels convert it through the vector, and the host converts it lane by
/// lane, since a host vector holds no lanes. Derived with
/// `#[device_repr(lanes)]` for a struct whose fields all have one scalar type.
pub trait LaneRepr: DeviceRepr<Layout = Packed, Repr = Vector<Self::Lane, Self::Width>> {
    type Lane: Scalar;
    /// Lanes per value.
    type Width: Size;

    /// The value's lanes, in field order.
    fn lanes(&self) -> impl Iterator<Item = Self::Lane>;

    /// The value whose lanes, in field order, are `lanes`.
    fn from_lanes(lanes: &[Self::Lane]) -> Self;
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

/// Values of `Self` order as their representations do, so device code may
/// compare them, or keep an atomic minimum or maximum, natively. Holds for
/// every ordered primitive; derived for brands with `#[device_repr(ord)]`.
pub trait OrderedRepr: DeviceRepr<Repr: CubePartialOrd> + PartialOrd {}

impl<T: CubePartialOrd> OrderedRepr for T {}

/// Same-brand `min`, `max` and `clamp` of an [`OrderedRepr`] brand,
/// computed on its representation. Derived with `#[device_repr(ord)]`;
/// primitives have their own.
pub trait Ordered: OrderedRepr
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
        let [lhs, rhs] = [self, other].map(|value| Self::Value::__expand_into_repr(scope, value));
        Self::Value::__expand_from_repr(scope, min::expand(scope, lhs, rhs))
    }

    fn __expand_max_method(self, scope: &Scope, other: Self) -> Self {
        let [lhs, rhs] = [self, other].map(|value| Self::Value::__expand_into_repr(scope, value));
        Self::Value::__expand_from_repr(scope, max::expand(scope, lhs, rhs))
    }

    fn __expand_clamp_method(self, scope: &Scope, min: Self, max: Self) -> Self {
        let [value, min, max] =
            [self, min, max].map(|value| Self::Value::__expand_into_repr(scope, value));
        Self::Value::__expand_from_repr(scope, clamp::expand(scope, value, min, max))
    }
}

/// Addresses the elements of a [`Storage`](crate::prelude::Storage), so a
/// table can accept only its own ID type. Derived for `u32` and `usize` brands
/// with `#[device_repr(key)]`; composite keys implement it with `#[cube]`.
#[cube]
pub trait StorageKey: CubeType + 'static {
    /// The position of the value this key addresses.
    fn position(key: Self) -> usize;
}

#[cube]
impl StorageKey for usize {
    fn position(key: usize) -> usize {
        key
    }
}

#[cube]
impl StorageKey for u32 {
    fn position(key: u32) -> usize {
        key as usize
    }
}

/// The same brand over another native representation, such as vectors of its
/// scalar. Derived when the wrapped field's type is a type parameter.
pub trait WithRepr<R: CubePrimitive>: DeviceRepr {
    type Output: DeviceRepr<Repr = R>;
}
