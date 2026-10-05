use core::{future::Future, marker::PhantomData, ops::Range};

use alloc::vec::Vec;
use cubecl_common::bytes::Bytes;
use cubecl_runtime::{
    client::Client,
    server::{Handle, ServerError},
};

use crate::{CubeElement, prelude::BufferArg};

/// A device allocation of `len` elements of `T`, which a kernel takes as a
/// slice without `unsafe`.
///
/// [`BufferArg::from_raw_parts`] is unsafe because the caller states the
/// length. A `Buffer` records it when it allocates, so turning it into a
/// [`BufferArg`] is safe. What a kernel can reach is bounded by the
/// allocation either way: a checked launch indexes against the length the
/// launcher derives from the handle's size, never against a declared one.
pub struct Buffer<T: CubeElement> {
    handle: Handle,
    len: usize,
    _element: PhantomData<T>,
}

impl<T: CubeElement> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.clone(),
            len: self.len,
            _element: PhantomData,
        }
    }
}

impl<T: CubeElement> core::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer")
            .field("element", &T::type_name())
            .field("len", &self.len)
            .finish()
    }
}

impl<T: CubeElement> Buffer<T> {
    /// A buffer holding a copy of `data`.
    pub fn create(client: &Client, data: &[T]) -> Self {
        Self {
            handle: client.create_from_slice(T::as_bytes(data)),
            len: data.len(),
            _element: PhantomData,
        }
    }

    /// A buffer of `len` elements whose contents are unspecified until a
    /// kernel writes them.
    pub fn empty(client: &Client, len: usize) -> Self {
        Self {
            handle: client.empty(len * size_of::<T>()),
            len,
            _element: PhantomData,
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The elements in `range`, sharing this buffer's memory.
    ///
    /// # Panics
    ///
    /// When `range` reaches past the end or is reversed, as slicing does.
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(
            range.start <= range.end && range.end <= self.len,
            "range {range:?} out of bounds for a buffer of {} elements",
            self.len
        );
        let size = size_of::<T>() as u64;
        let handle = self
            .handle
            .clone()
            .offset_start(range.start as u64 * size)
            .offset_end((self.len - range.end) as u64 * size);
        Self {
            handle,
            len: range.end - range.start,
            _element: PhantomData,
        }
    }

    /// Overwrites the first `data.len()` elements with `data`, in place and
    /// in order with the client's other work; [`Buffer::slice`] first to
    /// write elsewhere.
    ///
    /// # Panics
    ///
    /// When `data` is longer than the buffer.
    pub fn write(&self, client: &Client, data: &[T]) {
        assert!(
            data.len() <= self.len,
            "{} elements written to a buffer of {}",
            data.len(),
            self.len
        );
        if data.is_empty() {
            return;
        }
        let target = self.slice(0..data.len());
        client.write(
            &target.handle,
            Bytes::from_bytes_vec(T::as_bytes(data).to_vec()),
        );
    }

    /// Read the elements back once the device has written them, without
    /// blocking a thread: the read is queued now and resolves when its
    /// copy completes.
    pub fn read_async(
        &self,
        client: &Client,
    ) -> impl Future<Output = Result<Vec<T>, ServerError>> + Send + 'static + use<T> {
        let len = self.len;
        let read = client.read_async(alloc::vec![self.handle.clone()]);
        async move {
            let mut bytes = read.await?;
            Ok(T::from_bytes(&bytes.remove(0))[..len].to_vec())
        }
    }

    /// Read the elements back.
    pub fn read(&self, client: &Client) -> Result<Vec<T>, ServerError> {
        let bytes = client.read_one(self.handle.clone())?;
        Ok(T::from_bytes(&bytes)[..self.len].to_vec())
    }

    /// The handle, for the APIs that take one.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The handle, giving up the element type.
    pub fn into_handle(self) -> Handle {
        self.handle
    }
}

impl<T: CubeElement> From<&Buffer<T>> for BufferArg {
    fn from(buffer: &Buffer<T>) -> Self {
        // SAFETY: the length was recorded when the allocation was made for
        // exactly that many elements of `T`.
        unsafe { BufferArg::from_raw_parts(buffer.handle.clone(), buffer.len) }
            .typed(T::cube_type())
    }
}

impl<T: CubeElement> From<Buffer<T>> for BufferArg {
    fn from(buffer: Buffer<T>) -> Self {
        // SAFETY: as above.
        unsafe { BufferArg::from_raw_parts(buffer.handle, buffer.len) }.typed(T::cube_type())
    }
}
