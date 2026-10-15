use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

/// The keys of a value in a table: one [`StorageKey`], or a tuple of up to
/// four laid out row-major. A table keeps the extents of the axes after the
/// first, so a key tuple names one position.
#[cube]
pub trait Keys: CubeType + Send + Sync + 'static {
    /// The extents of the axes after the first.
    type Extents: CubeType<ExpandType: Clone> + Clone + Send + Sync + 'static;

    /// Where `keys` falls among the table's values.
    fn offset(keys: Self, extents: &Self::Extents) -> usize;
}

/// The one value of a table without axes.
#[cube]
impl Keys for () {
    type Extents = ();

    fn offset(_keys: (), _extents: &()) -> usize {
        0usize
    }
}

#[cube]
impl<A: StorageKey + Send + Sync> Keys for A {
    type Extents = ();

    fn offset(keys: A, _extents: &()) -> usize {
        A::position(keys)
    }
}

#[cube]
impl<A: StorageKey + Send + Sync, B: StorageKey + Send + Sync> Keys for (A, B) {
    type Extents = usize;

    fn offset(keys: (A, B), extents: &usize) -> usize {
        A::position(keys.0) * *extents + B::position(keys.1)
    }
}

#[cube]
impl<A: StorageKey + Send + Sync, B: StorageKey + Send + Sync, C: StorageKey + Send + Sync> Keys
    for (A, B, C)
{
    type Extents = (usize, usize);

    fn offset(keys: (A, B, C), extents: &(usize, usize)) -> usize {
        (A::position(keys.0) * extents.0 + B::position(keys.1)) * extents.1 + C::position(keys.2)
    }
}

#[cube]
impl<
    A: StorageKey + Send + Sync,
    B: StorageKey + Send + Sync,
    C: StorageKey + Send + Sync,
    D: StorageKey + Send + Sync,
> Keys for (A, B, C, D)
{
    type Extents = (usize, usize, usize);

    fn offset(keys: (A, B, C, D), extents: &(usize, usize, usize)) -> usize {
        ((A::position(keys.0) * extents.0 + B::position(keys.1)) * extents.1 + C::position(keys.2))
            * extents.2
            + D::position(keys.3)
    }
}

/// Values of `Q` addressed by keys `K`: one [`StorageKey`], or a tuple of
/// them over row-major axes, such as `(RowId, BucketId, PartId)`. A table of
/// a key it was not declared with does not compile.
#[derive(CubeType)]
pub struct Table<Q: DeviceRepr, V: SliceVisibility, K: Keys> {
    values: Storage<Q, V>,
    extents: K::Extents,
    #[cube(comptime)]
    keys: PhantomData<K>,
}

#[cube]
impl<Q: DeviceRepr, V: SliceVisibility, K: Keys> Table<Q, V, K> {
    /// Values the table holds.
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// The extents of the axes after the first.
    pub fn extents(&self) -> K::Extents {
        self.extents.clone()
    }

    pub fn load(&self, keys: K) -> Q {
        self.values.load(K::offset(keys, &self.extents))
    }

    /// The elements the values are stored as, row-major.
    pub fn elements(&self) -> &[StorageElement<Q>] {
        self.values.elements()
    }

    /// Where `keys` falls among the table's values.
    pub fn position(&self, keys: K) -> usize {
        K::offset(keys, &self.extents)
    }
}

#[cube]
impl<Q: DeviceRepr, V: SliceVisibility> Table<Q, V, ()> {
    /// The value of a table without axes.
    pub fn get(&self) -> Q {
        self.values.load(0usize)
    }
}

#[cube]
impl<Q: DeviceRepr, K: Keys> Table<Q, ReadWrite, K> {
    pub fn store(&mut self, keys: K, value: Q) {
        let at = K::offset(keys, &self.extents);
        self.values.store(at, value);
    }

    /// The elements the values are stored as, row-major, to write.
    pub fn elements_mut(&mut self) -> &mut [StorageElement<Q>] {
        self.values.elements_mut()
    }
}

impl<Q: DeviceRepr, V: SliceVisibility, K: Keys> Table<Q, V, K> {
    /// The table of `len` values of `buffer` from position `at`, with the
    /// extents of its axes after the first.
    #[doc(hidden)]
    pub fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<StorageElement<Q>>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <K::Extents as CubeType>::ExpandType,
    ) -> TableExpand<Q, V, K> {
        TableExpand {
            values: Storage::__expand_view(scope, buffer, at, len),
            extents,
            keys: PhantomData,
        }
    }
}

/// Values of `Q` addressed by keys `K`, updated atomically.
#[derive(CubeType)]
pub struct AtomicTable<Q: DeviceRepr, K: Keys> {
    values: AtomicStorage<Q>,
    extents: K::Extents,
    #[cube(comptime)]
    keys: PhantomData<K>,
}

#[cube]
impl<Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>>, K: Keys> AtomicTable<Q, K> {
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn load(&self, keys: K) -> Q {
        self.values.load(K::offset(keys, &self.extents))
    }

    pub fn store(&self, keys: K, value: Q) {
        self.values.store(K::offset(keys, &self.extents), value);
    }

    /// Stores `value`, returning the previous value.
    pub fn exchange(&self, keys: K, value: Q) -> Q {
        self.values.exchange(K::offset(keys, &self.extents), value)
    }
}

#[cube]
impl<Q, K: Keys> AtomicTable<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>> + core::ops::Add<Output = Q>,
{
    /// Adds `value`, returning the previous value.
    pub fn fetch_add(&self, keys: K, value: Q) -> Q {
        self.values.fetch_add(K::offset(keys, &self.extents), value)
    }
}

#[cube]
impl<Q, K: Keys> AtomicTable<Q, K>
where
    Q: OrderedRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>>,
{
    /// Keeps the smaller value, returning the previous value.
    pub fn fetch_min(&self, keys: K, value: Q) -> Q {
        self.values.fetch_min(K::offset(keys, &self.extents), value)
    }

    /// Keeps the larger value, returning the previous value.
    pub fn fetch_max(&self, keys: K, value: Q) -> Q {
        self.values.fetch_max(K::offset(keys, &self.extents), value)
    }
}

#[cube]
impl<Q, K: Keys> AtomicTable<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: Int>> + PartialEq,
{
    /// Stores `value` if the current value is `expected`, returning the
    /// previous value.
    pub fn compare_exchange_weak(&self, keys: K, expected: Q, value: Q) -> Q {
        self.values
            .compare_exchange_weak(K::offset(keys, &self.extents), expected, value)
    }
}

impl<Q: DeviceRepr, K: Keys> AtomicTable<Q, K> {
    /// The table of `len` values of `buffer` from position `at`, with the
    /// extents of its axes after the first.
    #[doc(hidden)]
    pub fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<Atomic<Q::Repr>>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <K::Extents as CubeType>::ExpandType,
    ) -> AtomicTableExpand<Q, K> {
        AtomicTableExpand {
            values: AtomicStorage::__expand_view(scope, buffer, at, len),
            extents,
            keys: PhantomData,
        }
    }
}
