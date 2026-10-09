use crate::prelude::*;

/// A host value a kernel reads as `Device`: a resident buffer set, or a
/// description lowered to one. `#[derive(Resident)]` implements it for the
/// owned mirror of a launchable struct.
pub trait Bind {
    type Device: LaunchArg;

    /// The value as a kernel argument.
    fn arg(&self) -> <Self::Device as LaunchArg>::RuntimeArg;
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
