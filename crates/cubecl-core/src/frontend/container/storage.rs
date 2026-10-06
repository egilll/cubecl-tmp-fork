use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

/// Device storage of `Q` values in their native representation, addressed by
/// `K`. A table keyed by an ID brand accepts only that ID.
#[derive(CubeType)]
pub struct Storage<Q: DeviceRepr, V: SliceVisibility = ReadWrite, K: StorageKey = usize> {
    inner: alloc::boxed::Box<[Q::Repr]>,
    #[cube(comptime)]
    marker: PhantomData<(Q, V, K)>,
}

#[cube]
impl<Q: DeviceRepr, V: SliceVisibility, K: StorageKey> Storage<Q, V, K> {
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn load(&self, key: K) -> Q {
        intrinsic!(|scope| { self.__expand_load_at_method(scope, K::__expand_position(scope, key)) })
    }

    fn load_at(&self, position: usize) -> Q {
        intrinsic!(|scope| {
            Q::expand_from_repr(
                self.inner
                    .__expand_index_method(scope, position)
                    .__expand_deref_method(scope),
            )
        })
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
        intrinsic!(|scope| {
            let value = Q::expand_into_repr(value);
            self.inner
                .__expand_index_mut_method(scope, position)
                .__expand_assign_method(scope, value);
        })
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

    pub fn from_native(buffer: &Buffer<Q::Repr>) -> Self
    where
        Q::Repr: CubeElement,
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
        <[Q::Repr]>::register(arg.inner, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> Self::ExpandType {
        StorageExpand {
            inner: <alloc::boxed::Box<[Q::Repr]>>::expand(arg, builder),
            marker: PhantomData,
        }
    }
}
