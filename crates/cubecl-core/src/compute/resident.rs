use crate::prelude::*;

/// A host value a kernel reads as `Device`: a resident buffer set, or a
/// description lowered to one. `#[derive(Resident)]` implements it for the
/// owned mirror of a launchable struct.
pub trait Bind {
    type Device: LaunchArg;

    /// The value as a kernel argument.
    fn arg(&self) -> <Self::Device as LaunchArg>::RuntimeArg;
}

/// One buffer of a resident value, sized before it exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferSize {
    /// The field it backs.
    pub name: &'static str,
    /// Values it holds: at least one, so an empty table still binds.
    pub elements: usize,
    pub element_bytes: usize,
}

impl BufferSize {
    /// A buffer of `elements` values of `Q`, at least one.
    #[must_use]
    pub fn of<Q: DeviceRepr>(name: &'static str, elements: usize) -> Self
    where
        StorageElement<Q>: CubeElement,
    {
        Self {
            name,
            elements: elements.max(1),
            element_bytes: size_of::<StorageElement<Q>>(),
        }
    }

    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.elements * self.element_bytes
    }
}

/// Where a resident value's buffers come from: fresh allocations, or one
/// allocation carved in the order a shape lists its buffers.
pub trait Carve {
    /// The buffer `size` describes, holding values of `Q`; its contents
    /// are unspecified until written.
    fn carve<Q: DeviceRepr>(&mut self, size: BufferSize) -> StorageBuffer<Q>
    where
        StorageElement<Q>: CubeElement;
}

/// A launchable value with an owned host mirror, as `#[derive(Resident)]`
/// declares it.
pub trait Residency: LaunchArg {
    /// The owned host mirror.
    type Resident: Bind<Device = Self>;
}

/// A resident value whose buffers can be sized and carved before any data
/// exists.
pub trait Shaped: Sized {
    /// Every buffer's size and every scalar's value.
    type Shape: Clone;

    /// Visits each buffer's size, in carving order.
    fn sizes(shape: &Self::Shape, visit: &mut dyn FnMut(BufferSize));

    /// The bytes every buffer holds together.
    fn bytes(shape: &Self::Shape) -> usize {
        let mut bytes = 0;
        Self::sizes(shape, &mut |size| bytes += size.bytes());
        bytes
    }

    /// The value with every buffer carved from `carver` in the order
    /// [`Self::sizes`] lists them.
    fn carve(shape: &Self::Shape, carver: &mut impl Carve) -> Self;
}

/// The grid of a shape: rows of `width` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GridShape {
    pub rows: usize,
    pub width: u32,
}

/// Device memory a resident value holds, in bytes.
pub trait ResidentBytes {
    fn bytes(&self) -> usize;
}

impl<Q: DeviceRepr> ResidentBytes for StorageBuffer<Q>
where
    StorageElement<Q>: CubeElement,
{
    fn bytes(&self) -> usize {
        self.as_native().len() * size_of::<StorageElement<Q>>()
    }
}

/// The owned host side of a [`Grid`]: its values and row width.
#[derive(Clone)]
pub struct ResidentGrid<Q: DeviceRepr>
where
    StorageElement<Q>: CubeElement,
{
    pub values: StorageBuffer<Q>,
    pub width: u32,
}

impl<Q: DeviceRepr + Send + Sync> ResidentGrid<Q>
where
    StorageElement<Q>: CubeElement,
{
    /// As a kernel argument keyed by `R` and `C`.
    #[must_use]
    pub fn arg<V: SliceVisibility, R: StorageKey + Send + Sync, C: StorageKey + Send + Sync>(
        &self,
    ) -> GridLaunch<Q, V, R, C> {
        GridLaunch::new((&self.values).into(), self.width)
    }
}
