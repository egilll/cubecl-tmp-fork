use alloc::vec::Vec;
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
        let values: Vec<_> = values.iter().cloned().map(Q::into_repr).collect();
        Self {
            inner: Buffer::create(client, &values),
            marker: PhantomData,
        }
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
        let values: Vec<_> = values.iter().cloned().map(Q::into_repr).collect();
        self.inner.write(client, &values);
    }

    pub fn read(&self, client: &Client) -> Result<Vec<Q>, ServerError> {
        self.inner
            .read(client)
            .map(|values| values.into_iter().map(Q::from_repr).collect())
    }

    pub fn read_async(
        &self,
        client: &Client,
    ) -> impl core::future::Future<Output = Result<Vec<Q>, ServerError>> + use<Q> {
        let read = self.inner.read_async(client);
        async move {
            read.await
                .map(|values| values.into_iter().map(Q::from_repr).collect())
        }
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

impl<Q: DeviceRepr<Repr: CubeElement>> From<&StorageBuffer<Q>> for TypedBufferArg<Q> {
    fn from(value: &StorageBuffer<Q>) -> Self {
        Self::from_native(&value.inner)
    }
}
