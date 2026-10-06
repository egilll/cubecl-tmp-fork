use crate::prelude::{CubePrimitive, CubeType, Scope};

/// A value stored on the device as a native primitive, such as a unit or ID
/// brand over `f32` or `u32`.
///
/// Every `Repr` value must be a valid `Self`: storage loads and host reads
/// rebuild values with [`from_repr`](Self::from_repr) and never validate them.
/// A refinement such as a positive distance stores its unrestricted quantity
/// instead and is checked where it's constructed.
pub trait DeviceRepr: CubeType + Sized + 'static {
    type Repr: CubePrimitive;

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
