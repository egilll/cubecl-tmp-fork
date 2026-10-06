use alloc::{borrow::Cow, vec::Vec};
use core::{marker::PhantomData, ops::Range};

use crate::prelude::*;
use cubecl_runtime::server::ServerError;

pub struct StorageBuffer<Q: DeviceRepr<Repr: CubeElement>> {
    inner: Buffer<Q::Repr>,
    marker: PhantomData<Q>,
}

impl<Q: DeviceRepr<Repr: CubeElement>> Clone for StorageBuffer<Q> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<Q: DeviceRepr<Repr: CubeElement>> StorageBuffer<Q> {
    pub fn create(client: &Client, values: &[Q]) -> Self
    where
        Q: Clone,
    {
        Self::from_native(Buffer::create(client, &reprs(values)))
    }

    pub fn empty(client: &Client, len: usize) -> Self {
        Self {
            inner: Buffer::empty(client, len),
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn slice(&self, range: Range<usize>) -> Self {
        Self {
            inner: self.inner.slice(range),
            marker: PhantomData,
        }
    }

    pub fn write(&self, client: &Client, values: &[Q])
    where
        Q: Clone,
    {
        self.inner.write(client, &reprs(values));
    }

    pub fn read(&self, client: &Client) -> Result<Vec<Q>, ServerError> {
        self.inner.read(client).map(values)
    }

    pub fn read_async(
        &self,
        client: &Client,
    ) -> impl core::future::Future<Output = Result<Vec<Q>, ServerError>> + use<Q> {
        let read = self.inner.read_async(client);
        async move { read.await.map(values) }
    }

    /// Binds these values to a kernel storage of `N`-lane vectors with the
    /// same brand, such as `Quantity<Tag, Vector<f32, Const<4>>>` for a buffer
    /// of `Quantity<Tag, f32>`.
    ///
    /// # Panics
    ///
    /// When `N` isn't a power of two, or the buffer's length or offset isn't
    /// a whole number of vectors.
    pub fn vectors<const N: usize>(
        &self,
    ) -> TypedBufferArg<<Q as WithRepr<Vector<<Q as DeviceRepr>::Repr, Const<N>>>>::Output>
    where
        Q: WithRepr<Vector<<Q as DeviceRepr>::Repr, Const<N>>>,
        <Q as DeviceRepr>::Repr: Scalar,
    {
        let vector = N * size_of::<Q::Repr>();
        let offset = self.inner.handle().offset_start.unwrap_or(0) as usize;
        assert!(N.is_power_of_two(), "{N}-lane vectors are padded in storage");
        assert!(
            self.len() % N == 0 && offset % vector == 0,
            "{} values at byte {offset} aren't whole {N}-lane vectors",
            self.len()
        );
        TypedBufferArg::from_arg(self.as_native().into())
    }

    pub fn as_native(&self) -> &Buffer<Q::Repr> {
        &self.inner
    }

    pub fn from_native(inner: Buffer<Q::Repr>) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

fn reprs<Q: DeviceRepr + Clone>(values: &[Q]) -> Cow<'_, [Q::Repr]> {
    match Q::TRANSPARENT {
        Some(proof) => Cow::Borrowed(proof.reprs(values)),
        None => Cow::Owned(values.iter().cloned().map(Q::into_repr).collect()),
    }
}

fn values<Q: DeviceRepr>(reprs: Vec<Q::Repr>) -> Vec<Q> {
    match Q::TRANSPARENT {
        Some(proof) => proof.values(reprs),
        None => reprs.into_iter().map(Q::from_repr).collect(),
    }
}

impl<Q: DeviceRepr<Repr: CubeElement>> From<&StorageBuffer<Q>> for TypedBufferArg<Q> {
    fn from(value: &StorageBuffer<Q>) -> Self {
        Self::from_native(&value.inner)
    }
}
