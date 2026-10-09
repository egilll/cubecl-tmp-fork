use alloc::{borrow::Cow, vec::Vec};
use core::{marker::PhantomData, ops::Range};

use crate::prelude::*;
use cubecl_runtime::server::ServerError;

/// A device buffer of `Q` values in their storage layout.
pub struct StorageBuffer<Q: DeviceRepr>
where
    StorageElement<Q>: CubeElement,
{
    inner: Buffer<StorageElement<Q>>,
    marker: PhantomData<Q>,
}

impl<Q: DeviceRepr> Clone for StorageBuffer<Q>
where
    StorageElement<Q>: CubeElement,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            marker: PhantomData,
        }
    }
}

impl<Q: DeviceRepr> core::fmt::Debug for StorageBuffer<Q>
where
    StorageElement<Q>: CubeElement,
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("StorageBuffer")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<Q: DeviceRepr> StorageBuffer<Q>
where
    StorageElement<Q>: CubeElement,
{
    pub fn empty(client: &Client, len: usize) -> Self {
        Self::from_native(Buffer::empty(client, len * width::<Q>()))
    }

    /// Values from their elements in storage layout, such as three floats per
    /// packed point.
    ///
    /// # Panics
    ///
    /// When `elements` isn't a whole number of values.
    pub fn from_elements(client: &Client, elements: &[StorageElement<Q>]) -> Self {
        Self::from_native(Buffer::create(client, whole::<Q>(elements)))
    }

    pub fn write_elements(&self, client: &Client, elements: &[StorageElement<Q>]) {
        self.inner.write(client, whole::<Q>(elements));
    }

    pub fn read_elements(&self, client: &Client) -> Result<Vec<StorageElement<Q>>, ServerError> {
        self.inner.read(client)
    }

    pub fn read_elements_async(
        &self,
        client: &Client,
    ) -> impl core::future::Future<Output = Result<Vec<StorageElement<Q>>, ServerError>> + use<Q>
    {
        self.inner.read_async(client)
    }

    /// Values stored.
    pub fn len(&self) -> usize {
        self.inner.len() / width::<Q>()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The values in `range`.
    pub fn slice(&self, range: Range<usize>) -> Self {
        let width = width::<Q>();
        Self::from_native(self.inner.slice(range.start * width..range.end * width))
    }

    /// The buffer of storage elements.
    pub fn as_native(&self) -> &Buffer<StorageElement<Q>> {
        &self.inner
    }

    pub fn from_native(inner: Buffer<StorageElement<Q>>) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
}

impl<Q: DeviceRepr<Layout = Native, Repr: CubeElement>> StorageBuffer<Q> {
    pub fn create(client: &Client, values: &[Q]) -> Self
    where
        Q: Clone,
    {
        Self::from_native(Buffer::create(client, &reprs(values)))
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
        assert!(
            N.is_power_of_two(),
            "{N}-lane vectors are padded in storage"
        );
        assert!(
            self.len() % N == 0 && offset % vector == 0,
            "{} values at byte {offset} aren't whole {N}-lane vectors",
            self.len()
        );
        TypedBufferArg::from_arg(self.as_native().into())
    }
}

/// Host values of lane structs ([`LaneRepr`]), converted lane by lane.
impl<Q: LaneRepr> StorageBuffer<Q>
where
    Q::Lane: CubeElement,
{
    pub fn from_values(client: &Client, values: &[Q]) -> Self {
        Self::from_elements(client, &lanes(values))
    }

    pub fn write_values(&self, client: &Client, values: &[Q]) {
        self.write_elements(client, &lanes(values));
    }

    pub fn read_values(&self, client: &Client) -> Result<Vec<Q>, ServerError> {
        self.read_elements(client)
            .map(|elements| from_lanes(&elements))
    }

    pub fn read_values_async(
        &self,
        client: &Client,
    ) -> impl core::future::Future<Output = Result<Vec<Q>, ServerError>> + use<Q> {
        let read = self.read_elements_async(client);
        async move { read.await.map(|elements| from_lanes(&elements)) }
    }
}

fn lanes<Q: LaneRepr>(values: &[Q]) -> Vec<Q::Lane> {
    values.iter().flat_map(LaneRepr::lanes).collect()
}

fn from_lanes<Q: LaneRepr>(lanes: &[Q::Lane]) -> Vec<Q> {
    lanes
        .chunks_exact(Q::Width::value())
        .map(Q::from_lanes)
        .collect()
}

fn width<Q: DeviceRepr>() -> usize {
    StorageWidth::<Q>::value()
}

fn whole<Q: DeviceRepr>(elements: &[StorageElement<Q>]) -> &[StorageElement<Q>] {
    let width = width::<Q>();
    assert!(
        elements.len() % width == 0,
        "{} elements aren't whole values of {width}",
        elements.len()
    );
    elements
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

impl<Q: DeviceRepr> From<&StorageBuffer<Q>> for TypedBufferArg<Q>
where
    StorageElement<Q>: CubeElement,
{
    fn from(value: &StorageBuffer<Q>) -> Self {
        Self::from_native(&value.inner)
    }
}
