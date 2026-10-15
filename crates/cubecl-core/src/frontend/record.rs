//! Records: values stored as a fixed run of 32-bit words, every field in
//! words of its own, so a struct of mixed scalars, brands, enums and
//! nested records shares a buffer of `u32` with any other table.
//!
//! Every scalar a record holds is a record of one word: `u32` as itself,
//! `f32` and `i32` as their exact bits, `bool` as 0 or 1, and every
//! [`DeviceRepr`] over them (brands, kinds, niche options). Words are only
//! moved, never computed on, so every field round-trips losslessly.
//! `#[derive(Record)]` implements both sides for a struct whose fields are
//! records.

use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*};

/// A native scalar stored as one 32-bit word.
#[cube]
pub trait Word: CubePrimitive + Copy + Send + Sync + 'static {
    fn to_word(value: Self) -> u32;
    fn from_word(word: u32) -> Self;
}

#[cube]
impl Word for f32 {
    #[cube(inline)]
    fn to_word(value: f32) -> u32 {
        u32::reinterpret(value)
    }
    #[cube(inline)]
    fn from_word(word: u32) -> f32 {
        f32::reinterpret(word)
    }
}

#[cube]
impl Word for u32 {
    #[cube(inline)]
    fn to_word(value: u32) -> u32 {
        value
    }
    #[cube(inline)]
    fn from_word(word: u32) -> u32 {
        word
    }
}

#[cube]
impl Word for i32 {
    #[cube(inline)]
    fn to_word(value: i32) -> u32 {
        u32::reinterpret(value)
    }
    #[cube(inline)]
    fn from_word(word: u32) -> i32 {
        i32::reinterpret(word)
    }
}

#[cube]
impl Word for bool {
    #[cube(inline)]
    fn to_word(value: bool) -> u32 {
        select(value, 1u32, 0u32)
    }
    #[cube(inline)]
    fn from_word(word: u32) -> bool {
        word != 0
    }
}

/// The host's side of a [`Word`].
pub trait HostWord: Copy {
    fn encode(self) -> u32;
    fn decode(word: u32) -> Self;
}

impl HostWord for f32 {
    fn encode(self) -> u32 {
        self.to_bits()
    }
    fn decode(word: u32) -> Self {
        f32::from_bits(word)
    }
}

impl HostWord for u32 {
    fn encode(self) -> u32 {
        self
    }
    fn decode(word: u32) -> Self {
        word
    }
}

impl HostWord for i32 {
    fn encode(self) -> u32 {
        self.cast_unsigned()
    }
    fn decode(word: u32) -> Self {
        word.cast_signed()
    }
}

impl HostWord for bool {
    fn encode(self) -> u32 {
        self as u32
    }
    fn decode(word: u32) -> Self {
        word != 0
    }
}

/// A record's host form: its word count and how it writes and reads words.
pub trait Words: Sized {
    const WORDS: usize;
    fn write(&self, words: &mut [u32]);
    fn read(words: &[u32]) -> Self;

    /// Records laid out one after another.
    fn pack(records: &[Self]) -> Vec<u32> {
        let mut words = vec![0; records.len() * Self::WORDS];
        for (record, row) in records.iter().zip(words.chunks_mut(Self::WORDS.max(1))) {
            record.write(row);
        }
        words
    }

    /// Every record of a packed word buffer.
    // `as_chunks` would need `Self::WORDS` as a const generic argument.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn unpack(words: &[u32]) -> Vec<Self> {
        words.chunks_exact(Self::WORDS).map(Self::read).collect()
    }
}

/// A record's device form.
#[cube]
pub trait Record:
    Words
    + CubeType<ExpandType: Clone + Copy + Assign + RuntimeAssign>
    + Clone
    + Copy
    + Send
    + Sync
    + 'static
{
    /// The record whose first word is `words[at]`.
    fn read_at(words: &[u32], at: usize) -> Self;
    /// Writes the record from `words[at]` on.
    fn write_at(value: Self, words: &mut [u32], at: usize);

    /// Record `index` of a buffer of records.
    fn load(records: &[u32], index: usize) -> Self {
        let width = comptime![Self::WORDS];
        Self::read_at(records, index * width)
    }

    /// Overwrites record `index` of a buffer of records.
    fn store(value: Self, records: &mut [u32], index: usize) {
        let width = comptime![Self::WORDS];
        Self::write_at(value, records, index * width);
    }
}

impl<Q> Words for Q
where
    Q: DeviceRepr + Copy,
    Q::Repr: HostWord,
{
    const WORDS: usize = 1;

    fn write(&self, words: &mut [u32]) {
        words[0] = Q::into_repr(*self).encode();
    }

    fn read(words: &[u32]) -> Self {
        Q::from_repr(<Q::Repr as HostWord>::decode(words[0]))
    }
}

#[cube]
impl<Q> Record for Q
where
    Q: DeviceRepr
        + CubeType<ExpandType: Clone + Copy + Assign + RuntimeAssign>
        + Copy
        + Send
        + Sync,
    Q::Repr: Word + HostWord,
{
    #[cube(inline)]
    fn read_at(words: &[u32], at: usize) -> Q {
        Q::from_repr(Q::Repr::from_word(words[at]))
    }

    #[cube(inline)]
    fn write_at(value: Q, words: &mut [u32], at: usize) {
        words[at] = Q::Repr::to_word(Q::into_repr(value));
    }
}

/// Records of `R` addressed by keys `K`, in a buffer of words.
#[derive(CubeType)]
pub struct RecordTable<R: Record, V: SliceVisibility, K: Keys> {
    words: alloc::boxed::Box<[u32]>,
    extents: K::Extents,
    #[cube(comptime)]
    marker: PhantomData<(R, V, K)>,
}

#[cube]
impl<R: Record, V: SliceVisibility, K: Keys> RecordTable<R, V, K> {
    /// Records the table holds.
    pub fn len(&self) -> usize {
        self.words.len() / comptime![R::WORDS]
    }

    pub fn load(&self, keys: K) -> R {
        R::load(&self.words, K::offset(keys, &self.extents))
    }
}

#[cube]
impl<R: Record, V: SliceVisibility> RecordTable<R, V, ()> {
    /// The record of a table without axes.
    pub fn get(&self) -> R {
        R::load(&self.words, 0usize)
    }
}

#[cube]
impl<R: Record, K: Keys> RecordTable<R, ReadWrite, K> {
    pub fn store(&mut self, keys: K, value: R) {
        let at = K::offset(keys, &self.extents);
        R::store(value, &mut self.words, at);
    }
}

impl<R: Record, V: SliceVisibility, K: Keys> RecordTable<R, V, K> {
    /// The table of `len` words of `buffer` from word `at`, with the
    /// extents of its axes after the first.
    #[doc(hidden)]
    pub fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<u32>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <K::Extents as CubeType>::ExpandType,
    ) -> RecordTableExpand<R, V, K> {
        let list = buffer.__extract_list(scope);
        RecordTableExpand {
            words: crate::frontend::slice::from_raw_parts::<u32>(scope, list, at, len)
                .expand
                .into(),
            extents,
            marker: PhantomData,
        }
    }
}
