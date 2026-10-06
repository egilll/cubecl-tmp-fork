use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

/// Device storage of `Q` values in their native representation and layout,
/// addressed by `K`. A table keyed by an ID brand accepts only that ID.
#[derive(CubeType)]
pub struct Storage<Q: DeviceRepr, V: SliceVisibility = ReadWrite, K: StorageKey = usize> {
    inner: alloc::boxed::Box<[StorageElement<Q>]>,
    #[cube(comptime)]
    marker: PhantomData<(Q, V, K)>,
}

#[cube]
impl<Q: DeviceRepr, V: SliceVisibility, K: StorageKey> Storage<Q, V, K> {
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.inner.len() / StorageWidth::<Q>::value()
    }

    pub fn load(&self, key: K) -> Q {
        intrinsic!(|scope| { self.__expand_load_at_method(scope, K::__expand_position(scope, key)) })
    }

    fn load_at(&self, position: usize) -> Q {
        Q::from_repr(Q::Layout::load_from(&self.inner, position))
    }
}

#[cube]
impl<Q: DeviceRepr, K: StorageKey> Storage<Q, ReadWrite, K> {
    pub fn store(&mut self, key: K, value: Q) {
        intrinsic!(|scope| {
            self.__expand_store_at_method(scope, K::__expand_position(scope, key), value)
        })
    }

    pub fn fill(&mut self, value: Q) {
        let value = Q::into_repr(value);
        for position in 0..self.len() {
            self.store_at(position, Q::from_repr(value));
        }
    }

    /// Writes `table[keys[i]]` to position `i`.
    pub fn gather<T: DeviceRepr + StorageKey>(
        &mut self,
        table: &Storage<Q, ReadOnly, T>,
        keys: &Storage<T, ReadOnly>,
    ) {
        for position in 0..self.len() {
            self.store_at(position, table.load(keys.load_at(position)));
        }
    }

    fn store_at(&mut self, position: usize, value: Q) {
        Q::Layout::store_into(&mut self.inner, position, Q::into_repr(value));
    }
}

pub struct TypedBufferArg<Q: DeviceRepr> {
    inner: BufferArg,
    marker: PhantomData<Q>,
}

impl<Q: DeviceRepr> TypedBufferArg<Q> {
    pub fn alias(input_pos: usize, length: usize) -> Self {
        Self {
            inner: BufferArg::alias(input_pos, length),
            marker: PhantomData,
        }
    }

    pub fn from_native(buffer: &Buffer<StorageElement<Q>>) -> Self
    where
        StorageElement<Q>: CubeElement,
    {
        Self::from_arg(buffer.into())
    }

    pub(crate) fn from_arg(inner: BufferArg) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

impl<Q: DeviceRepr + Send + Sync, V: SliceVisibility, K: StorageKey + Send + Sync> LaunchArg
    for Storage<Q, V, K>
{
    type RuntimeArg = TypedBufferArg<Q>;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        launcher.register_storage_type::<Q>(&arg.inner);
        <[StorageElement<Q>]>::register(arg.inner, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Self::ExpandType {
        StorageExpand {
            inner: <alloc::boxed::Box<[StorageElement<Q>]>>::expand(arg, builder),
            marker: PhantomData,
        }
    }
}

/// Device storage of `Q` values updated atomically, addressed by `K`. Each
/// update needs the operation on `Q` it performs: `Add` for
/// [`fetch_add`](Self::fetch_add), an order for
/// [`fetch_max`](Self::fetch_max).
#[derive(CubeType)]
pub struct AtomicStorage<Q: DeviceRepr, K: StorageKey = usize> {
    inner: alloc::boxed::Box<[Atomic<Q::Repr>]>,
    #[cube(comptime)]
    marker: PhantomData<(Q, K)>,
}

#[cube]
impl<Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>>, K: StorageKey> AtomicStorage<Q, K> {
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn load(&self, key: K) -> Q {
        Q::from_repr(self.inner[K::position(key)].load())
    }

    pub fn store(&self, key: K, value: Q) {
        self.inner[K::position(key)].store(Q::into_repr(value));
    }

    /// Stores `value`, returning the previous value.
    pub fn exchange(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].exchange(Q::into_repr(value)))
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>> + core::ops::Add<Output = Q>,
{
    /// Adds `value`, returning the previous value.
    pub fn fetch_add(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_add(Q::into_repr(value)))
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>> + core::ops::Sub<Output = Q>,
{
    /// Subtracts `value`, returning the previous value.
    pub fn fetch_sub(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_sub(Q::into_repr(value)))
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: AtomicNumeric>> + PartialOrd,
{
    /// Keeps the smaller value, returning the previous value.
    pub fn fetch_min(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_min(Q::into_repr(value)))
    }

    /// Keeps the larger value, returning the previous value.
    pub fn fetch_max(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_max(Q::into_repr(value)))
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: Int>> + PartialEq,
{
    /// Stores `value` if the current value is `expected`, returning the
    /// previous value.
    pub fn compare_exchange_weak(&self, key: K, expected: Q, value: Q) -> Q {
        Q::from_repr(
            self.inner[K::position(key)]
                .compare_exchange_weak(Q::into_repr(expected), Q::into_repr(value)),
        )
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: Int>> + core::ops::BitOr<Output = Q>,
{
    /// Sets the bits of `value`, returning the previous value.
    pub fn fetch_or(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_or(Q::into_repr(value)))
    }
}

#[cube]
impl<Q, K: StorageKey> AtomicStorage<Q, K>
where
    Q: DeviceRepr<Repr: CubePrimitive<Scalar: Int>> + core::ops::BitAnd<Output = Q>,
{
    /// Keeps the bits of `value`, returning the previous value.
    pub fn fetch_and(&self, key: K, value: Q) -> Q {
        Q::from_repr(self.inner[K::position(key)].fetch_and(Q::into_repr(value)))
    }
}

impl<Q: DeviceRepr<Layout = Native> + Send + Sync, K: StorageKey + Send + Sync> LaunchArg
    for AtomicStorage<Q, K>
{
    type RuntimeArg = TypedBufferArg<Q>;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        launcher.register_storage_type::<Q>(&arg.inner);
        <[Atomic<Q::Repr>]>::register(arg.inner, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Self::ExpandType {
        AtomicStorageExpand {
            inner: <alloc::boxed::Box<[Atomic<Q::Repr>]>>::expand(arg, builder),
            marker: PhantomData,
        }
    }
}

/// Values of `Q` in registers, addressed by `K`: a typed [`Array`].
#[derive(CubeType)]
pub struct LocalStorage<Q: DeviceRepr, K: StorageKey = usize> {
    inner: Array<StorageElement<Q>>,
    #[cube(comptime)]
    marker: PhantomData<(Q, K)>,
}

#[cube]
impl<Q: DeviceRepr, K: StorageKey> LocalStorage<Q, K> {
    pub fn new(#[comptime] len: usize) -> Self {
        intrinsic!(|scope| {
            let width = StorageWidth::<Q>::__expand_value(scope);
            LocalStorageExpand {
                inner: Array::__expand_new(scope, len * width),
                marker: PhantomData,
            }
        })
    }

    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.inner.len() / StorageWidth::<Q>::value()
    }

    pub fn load(&self, key: K) -> Q {
        Q::from_repr(Q::Layout::load_from(self.inner.as_slice(), K::position(key)))
    }

    pub fn store(&mut self, key: K, value: Q) {
        Q::Layout::store_into(self.inner.as_mut_slice(), K::position(key), Q::into_repr(value));
    }
}
