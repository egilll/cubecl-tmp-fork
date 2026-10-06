use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

#[derive(CubeType)]
pub struct Storage<Q: DeviceRepr, V: SliceVisibility = ReadWrite> {
    inner: alloc::boxed::Box<[Q::Repr]>,
    #[cube(comptime)]
    marker: PhantomData<(Q, V)>,
}

#[cube]
impl<Q: DeviceRepr, V: SliceVisibility> Storage<Q, V> {
    #[cube(inline)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn load(&self, index: usize) -> Q {
        intrinsic!(|scope| {
            Q::expand_from_repr(
                self.inner
                    .__expand_index_method(scope, index)
                    .__expand_deref_method(scope),
            )
        })
    }
}

#[cube]
impl<Q: DeviceRepr> Storage<Q, ReadWrite> {
    pub fn fill(&mut self, value: Q) {
        let value = Q::into_repr(value);
        for index in 0..self.len() {
            self.store(index, Q::from_repr(value));
        }
    }

    pub fn gather(&mut self, table: &Storage<Q, ReadOnly>, indices: &[u32]) {
        for index in 0..self.len() {
            self.store(index, table.load(indices[index] as usize));
        }
    }

    pub fn store(&mut self, index: usize, value: Q) {
        intrinsic!(|scope| {
            let value = Q::expand_into_repr(value);
            self.inner
                .__expand_index_mut_method(scope, index)
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
        Self {
            inner: buffer.into(),
            marker: PhantomData,
        }
    }
}

impl<Q: DeviceRepr + Send + Sync, V: SliceVisibility> LaunchArg for Storage<Q, V> {
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
