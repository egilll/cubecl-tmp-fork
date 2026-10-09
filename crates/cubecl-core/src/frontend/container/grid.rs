use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

/// Values of `Q` over two axes, row-major: `R` picks a row of `width`
/// values and `C` a value in it. The grid carries its width, so keys are
/// plain pairs of indices.
#[derive(CubeType, CubeLaunch)]
pub struct Grid<
    Q: DeviceRepr + Send + Sync,
    V: SliceVisibility,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> {
    values: Storage<Q, V>,
    width: u32,
    #[cube(comptime)]
    axes: PhantomData<(R, C)>,
}

/// A read-only [`Grid`].
pub type GridTable<Q, R, C> = Grid<Q, ReadOnly, R, C>;
/// A writable [`Grid`].
pub type GridColumn<Q, R, C> = Grid<Q, ReadWrite, R, C>;

#[cube]
impl<
    Q: DeviceRepr + Send + Sync,
    V: SliceVisibility,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> Grid<Q, V, R, C>
{
    /// Values per row.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Rows held.
    pub fn rows(&self) -> u32 {
        self.values.len() as u32 / self.width
    }

    pub fn load(&self, row: R, column: C) -> Q {
        self.values
            .load(R::position(row) * self.width as usize + C::position(column))
    }
}

#[cube]
impl<Q: DeviceRepr + Send + Sync, R: StorageKey + Send + Sync, C: StorageKey + Send + Sync>
    Grid<Q, ReadWrite, R, C>
{
    pub fn store(&mut self, row: R, column: C, value: Q) {
        let at = R::position(row) * self.width as usize + C::position(column);
        self.values.store(at, value);
    }
}

/// Values of `Q` over two axes, row-major, updated atomically.
#[derive(CubeType, CubeLaunch)]
pub struct AtomicGrid<
    Q: DeviceRepr<Layout = Native> + Send + Sync,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> {
    values: AtomicStorage<Q>,
    width: u32,
    #[cube(comptime)]
    axes: PhantomData<(R, C)>,
}

#[cube]
impl<
    Q: DeviceRepr<Layout = Native, Repr: CubePrimitive<Scalar: AtomicNumeric>> + Send + Sync,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> AtomicGrid<Q, R, C>
{
    /// Values per row.
    pub fn width(&self) -> u32 {
        self.width
    }

    fn at(&self, row: R, column: C) -> usize {
        R::position(row) * self.width as usize + C::position(column)
    }

    pub fn load(&self, row: R, column: C) -> Q {
        self.values.load(self.at(row, column))
    }

    pub fn store(&self, row: R, column: C, value: Q) {
        self.values.store(self.at(row, column), value);
    }
}

#[cube]
impl<
    Q: DeviceRepr<Layout = Native, Repr: CubePrimitive<Scalar: AtomicNumeric>>
        + core::ops::Add<Output = Q>
        + Send
        + Sync,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> AtomicGrid<Q, R, C>
{
    /// Adds `value`, returning the previous value.
    pub fn fetch_add(&self, row: R, column: C, value: Q) -> Q {
        let at = R::position(row) * self.width as usize + C::position(column);
        self.values.fetch_add(at, value)
    }
}

#[cube]
impl<
    Q: OrderedRepr<Layout = Native, Repr: CubePrimitive<Scalar: AtomicNumeric>> + Send + Sync,
    R: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
> AtomicGrid<Q, R, C>
{
    /// Keeps the smaller value, returning the previous value.
    pub fn fetch_min(&self, row: R, column: C, value: Q) -> Q {
        let at = R::position(row) * self.width as usize + C::position(column);
        self.values.fetch_min(at, value)
    }

    /// Keeps the larger value, returning the previous value.
    pub fn fetch_max(&self, row: R, column: C, value: Q) -> Q {
        let at = R::position(row) * self.width as usize + C::position(column);
        self.values.fetch_max(at, value)
    }
}
