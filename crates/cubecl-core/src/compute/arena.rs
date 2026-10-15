//! Arenas: many tables in one binding per element type.
//!
//! A struct of tables (`Table`, `AtomicTable`, `RecordTable`) with
//! `#[derive(Arena)]` binds as one kernel argument, its tables of one buffer
//! element sharing one buffer, each a view of its own region, so a kernel
//! may take far more tables than a backend has bindings. Tables are grouped
//! by element type in declaration order, so the binding count is the number
//! of element types, not of tables.
//!
//! On the host, `SArena` (for a struct `S`) packs and uploads the tables
//! from an `SData` of sources, one per table, and keeps each table's region
//! as an [`ArenaColumn`] for reads and writes.

use alloc::{vec, vec::Vec};
use core::any::TypeId;
use core::marker::PhantomData;

use crate::{self as cubecl, prelude::*, server::Handle};
use cubecl_ir::ExpandValue;

/// The extents of a table's axes after the first, as a kernel receives them.
pub trait Extents: CubeType + Clone + Send + Sync + 'static {
    /// Axes after the first.
    const AXES: usize;

    /// The extents, as kernel scalars registered one per axis.
    fn expand_extents(builder: &mut KernelBuilder) -> <Self as CubeType>::ExpandType;
}

impl Extents for () {
    const AXES: usize = 0;

    fn expand_extents(_builder: &mut KernelBuilder) {}
}

impl Extents for usize {
    const AXES: usize = 1;

    fn expand_extents(builder: &mut KernelBuilder) -> NativeExpand<usize> {
        <usize as LaunchArg>::expand(&(), builder)
    }
}

impl Extents for (usize, usize) {
    const AXES: usize = 2;

    fn expand_extents(builder: &mut KernelBuilder) -> <Self as CubeType>::ExpandType {
        (<usize as LaunchArg>::expand(&(), builder), <usize as LaunchArg>::expand(&(), builder))
    }
}

impl Extents for (usize, usize, usize) {
    const AXES: usize = 3;

    fn expand_extents(builder: &mut KernelBuilder) -> <Self as CubeType>::ExpandType {
        let a = <usize as LaunchArg>::expand(&(), builder);
        let b = <usize as LaunchArg>::expand(&(), builder);
        (a, b, <usize as LaunchArg>::expand(&(), builder))
    }
}

/// A table an arena holds: a region of the buffer of its element.
pub trait ArenaTable: CubeType + Send + Sync + 'static {
    /// The kernel's element of the shared buffer.
    type Element: CubePrimitive;
    /// The element as the host writes and reads it.
    type Host: CubeElement + CubePrimitive + LaunchArg<RuntimeArg = Self::Host> + Default + PartialEq + Send + Sync;
    /// The extents of the axes after the first.
    type Extents: Extents;

    /// The table over `len` elements of `buffer` from element `at`.
    fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<Self::Element>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <Self::Extents as CubeType>::ExpandType,
    ) -> Self::ExpandType;
}

impl<Q: DeviceRepr + Send + Sync, V: SliceVisibility, K: Keys<Extents: Extents>> ArenaTable
    for Table<Q, V, K>
where
    StorageElement<Q>: CubeElement + LaunchArg<RuntimeArg = StorageElement<Q>> + Default + PartialEq + Send + Sync,
{
    type Element = StorageElement<Q>;
    type Host = StorageElement<Q>;
    type Extents = K::Extents;

    fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<Self::Element>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <Self::Extents as CubeType>::ExpandType,
    ) -> Self::ExpandType {
        Table::__expand_view(scope, buffer, at, len, extents)
    }
}

impl<Q: DeviceRepr + Send + Sync, K: Keys<Extents: Extents>> ArenaTable for AtomicTable<Q, K>
where
    Q::Repr: CubeElement + LaunchArg<RuntimeArg = Q::Repr> + Default + PartialEq + Send + Sync,
{
    type Element = Atomic<Q::Repr>;
    type Host = Q::Repr;
    type Extents = K::Extents;

    fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<Self::Element>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <Self::Extents as CubeType>::ExpandType,
    ) -> Self::ExpandType {
        AtomicTable::__expand_view(scope, buffer, at, len, extents)
    }
}

impl<R: Record, V: SliceVisibility, K: Keys<Extents: Extents>> ArenaTable
    for RecordTable<R, V, K>
{
    type Element = u32;
    type Host = u32;
    type Extents = K::Extents;

    fn __expand_view(
        scope: &Scope,
        buffer: &SliceExpand<Self::Element>,
        at: NativeExpand<usize>,
        len: NativeExpand<usize>,
        extents: <Self::Extents as CubeType>::ExpandType,
    ) -> Self::ExpandType {
        RecordTable::__expand_view(scope, buffer, at, len, extents)
    }
}

/// One shared buffer of an arena.
#[derive(Clone)]
struct SharedBuffer {
    handle: Handle,
    /// Elements it holds (at least one, so it binds).
    len: usize,
    element: TypeId,
    register: fn(BufferArg, &mut KernelLauncher) -> BufferCompilationArg,
    binding: fn(&Handle, usize) -> BufferArg,
}

/// Where a table lies in its arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArenaPlacement {
    /// Its shared buffer, in first-use order.
    pub buffer: usize,
    /// Its first element there, and its elements.
    pub at: usize,
    pub len: usize,
    /// The extents of the axes after the first.
    pub trailing: Vec<usize>,
}

/// An arena as a kernel argument: its shared buffers and every table's
/// placement, in declaration order.
pub struct ArenaArg {
    buffers: Vec<(BufferArg, fn(BufferArg, &mut KernelLauncher) -> BufferCompilationArg)>,
    tables: Vec<(usize, usize, usize, Vec<usize>)>,
}

/// Which shared buffer each table of a launched arena views.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArenaCompilationArg {
    buffers: Vec<BufferCompilationArg>,
    tables: Vec<usize>,
}

impl ArenaArg {
    /// Registers the shared buffers, then each table's placement.
    pub fn register(self, launcher: &mut KernelLauncher) -> ArenaCompilationArg {
        let buffers = self.buffers.into_iter().map(|(arg, register)| register(arg, launcher)).collect();
        let mut tables = Vec::with_capacity(self.tables.len());
        for (buffer, at, len, trailing) in self.tables {
            <usize as LaunchArg>::register(at, launcher);
            <usize as LaunchArg>::register(len, launcher);
            for extent in trailing {
                <usize as LaunchArg>::register(extent, launcher);
            }
            tables.push(buffer);
        }
        ArenaCompilationArg { buffers, tables }
    }
}

/// The kernel side of a launched arena: its tables, expanded in declaration
/// order, each buffer bound once, at its first table.
pub struct ArenaExpansion<'a> {
    arg: &'a ArenaCompilationArg,
    buffers: Vec<Option<ExpandValue>>,
    next: usize,
}

impl<'a> ArenaExpansion<'a> {
    pub fn new(arg: &'a ArenaCompilationArg) -> Self {
        Self { arg, buffers: vec![None; arg.buffers.len()], next: 0 }
    }

    /// The next table.
    pub fn table<T: ArenaTable>(&mut self, builder: &mut KernelBuilder) -> T::ExpandType {
        let index = self.arg.tables[self.next];
        self.next += 1;
        let value = match &self.buffers[index] {
            Some(value) => value.clone(),
            None => {
                let value = <[T::Element]>::expand(&self.arg.buffers[index], builder).expand;
                self.buffers[index] = Some(value.clone());
                value
            }
        };
        let buffer: SliceExpand<T::Element> = value.into();
        let at = <usize as LaunchArg>::expand(&(), builder);
        let len = <usize as LaunchArg>::expand(&(), builder);
        let extents = <T::Extents as Extents>::expand_extents(builder);
        T::__expand_view(&builder.scope, &buffer, at, len, extents)
    }
}

/// What a table initially holds, and its shape (first axis first).
#[derive(Clone, Copy, Debug)]
pub struct ArenaSource<'a, E> {
    content: Content<'a, E>,
    trailing: [usize; 3],
    axes: usize,
}

#[derive(Clone, Copy, Debug)]
enum Content<'a, E> {
    Values(&'a [E]),
    Fill(E, usize),
    Repeat(&'a [E], usize),
}

impl<'a, E: Copy + Default> ArenaSource<'a, E> {
    /// `values`, as one axis.
    pub fn values(values: &'a [E]) -> Self {
        Self { content: Content::Values(values), trailing: [0; 3], axes: 1 }
    }

    /// `len` copies of `value`, as one axis.
    pub fn fill(value: E, len: usize) -> Self {
        Self { content: Content::Fill(value, len), trailing: [0; 3], axes: 1 }
    }

    /// Copies of a block of elements up to `len`, as one axis.
    pub fn repeat(block: &'a [E], len: usize) -> Self {
        assert!(!block.is_empty() && len.is_multiple_of(block.len()), "the block must divide the table");
        Self { content: Content::Repeat(block, len), trailing: [0; 3], axes: 1 }
    }

    /// Default elements of a shape.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::fill(E::default(), shape.iter().product()).shaped(shape)
    }

    /// The same contents viewed with `shape` (first axis first): up to four
    /// axes whose product is the length.
    pub fn shaped(mut self, shape: &[usize]) -> Self {
        assert!((1..=4).contains(&shape.len()), "a table has one to four axes");
        assert_eq!(shape.iter().product::<usize>(), self.len(), "the shape must cover the table");
        self.trailing = [0; 3];
        self.trailing[..shape.len() - 1].copy_from_slice(&shape[1..]);
        self.axes = shape.len();
        self
    }

    /// Elements the table holds.
    pub fn len(&self) -> usize {
        match self.content {
            Content::Values(values) => values.len(),
            Content::Fill(_, len) | Content::Repeat(_, len) => len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn trailing(&self) -> &[usize] {
        &self.trailing[..self.axes - 1]
    }
}

/// Tables of one element type start at multiples of this many bytes, so a
/// table's region can also be bound on its own.
pub const ARENA_ALIGN: usize = 256;

/// The host side of an arena's buffer of one element type while it is
/// packed.
struct Packing {
    element: TypeId,
    element_size: usize,
    bytes: Vec<u8>,
    fills: Vec<(usize, usize, Vec<u8>)>,
    any_values: bool,
    register: fn(BufferArg, &mut KernelLauncher) -> BufferCompilationArg,
    binding: fn(&Handle, usize) -> BufferArg,
    fill: fn(&Client, &Handle, usize, usize, &[u8]),
}

/// Packs an arena's tables: tables of one element type share a buffer, in
/// first-use order.
#[derive(Default)]
pub struct ArenaBuilder {
    buffers: Vec<Packing>,
    tables: Vec<ArenaPlacement>,
}

impl ArenaBuilder {
    /// Places the next table.
    pub fn table<T: ArenaTable>(&mut self, source: ArenaSource<'_, T::Host>) {
        assert_eq!(
            source.trailing().len(),
            <T::Extents as Extents>::AXES,
            "a table's source gives each axis after the first an extent"
        );
        let element = TypeId::of::<T::Element>();
        let index = match self.buffers.iter().position(|buffer| buffer.element == element) {
            Some(index) => index,
            None => {
                self.buffers.push(Packing {
                    element,
                    element_size: size_of::<T::Host>(),
                    bytes: Vec::new(),
                    fills: Vec::new(),
                    any_values: false,
                    register: <[T::Element]>::register,
                    binding: |handle, len| Buffer::<T::Host>::from_handle(handle.clone(), len).into(),
                    fill: fill_region::<T::Host>,
                });
                self.buffers.len() - 1
            }
        };
        let packing = &mut self.buffers[index];
        let size = packing.element_size;
        let align = ARENA_ALIGN / size;
        let at = (packing.bytes.len() / size).next_multiple_of(align);
        packing.bytes.resize(at * size, 0);
        let len = source.len();
        match source.content {
            Content::Values(values) => {
                packing.bytes.extend_from_slice(T::Host::as_bytes(values));
                packing.any_values |= !values.is_empty();
            }
            Content::Fill(value, len) => {
                let one = T::Host::as_bytes(core::slice::from_ref(&value)).to_vec();
                for _ in 0..len {
                    packing.bytes.extend_from_slice(&one);
                }
                if value != T::Host::default() {
                    packing.fills.push((at, len, one));
                }
            }
            Content::Repeat(block, len) => {
                for chunk in core::iter::repeat_n(T::Host::as_bytes(block), len / block.len()) {
                    packing.bytes.extend_from_slice(chunk);
                }
                packing.any_values = true;
            }
        }
        self.tables.push(ArenaPlacement { buffer: index, at, len, trailing: source.trailing().to_vec() });
    }

    /// The arena: each buffer uploaded when any of its tables has values,
    /// otherwise allocated and filled on the device.
    pub fn finish(self, client: &Client) -> Arena {
        let buffers = self
            .buffers
            .into_iter()
            .map(|packing| {
                let len = (packing.bytes.len() / packing.element_size).max(1);
                let handle = if packing.any_values {
                    let mut bytes = packing.bytes;
                    bytes.resize(len * packing.element_size, 0);
                    client.create_from_slice(&bytes)
                } else {
                    let handle = client.empty(len * packing.element_size);
                    (packing.fill)(client, &handle, 0, len, &vec![0; packing.element_size]);
                    for (at, len, value) in &packing.fills {
                        (packing.fill)(client, &handle, *at, *len, value);
                    }
                    handle
                };
                SharedBuffer { handle, len, element: packing.element, register: packing.register, binding: packing.binding }
            })
            .collect();
        Arena { buffers, tables: self.tables }
    }
}

/// The host side of an arena: its shared buffers and where its tables lie.
#[derive(Clone)]
pub struct Arena {
    buffers: Vec<SharedBuffer>,
    tables: Vec<ArenaPlacement>,
}

impl Arena {
    /// Table `index`'s region and shape.
    pub fn column<E: CubeElement>(&self, index: usize) -> ArenaColumn<E> {
        let placement = &self.tables[index];
        let buffer = &self.buffers[placement.buffer];
        let whole = Buffer::<E>::from_handle(buffer.handle.clone(), buffer.len);
        let mut shape = vec![placement.len / placement.trailing.iter().product::<usize>().max(1)];
        shape.extend(&placement.trailing);
        ArenaColumn { buffer: whole.slice(placement.at..placement.at + placement.len), shape, placement: placement.clone() }
    }

    /// Where each table lies.
    pub fn placements(&self) -> &[ArenaPlacement] {
        &self.tables
    }

    /// The shared buffers, as the bytes each holds.
    pub fn bytes(&self) -> usize {
        self.buffers.iter().map(|buffer| buffer.handle.size_in_used() as usize).sum()
    }

    /// The arena as a kernel argument.
    pub fn arg(&self) -> ArenaArg {
        ArenaArg {
            buffers: self.buffers.iter().map(|buffer| ((buffer.binding)(&buffer.handle, buffer.len), buffer.register)).collect(),
            tables: self.tables.iter().map(|table| (table.buffer, table.at, table.len, table.trailing.clone())).collect(),
        }
    }
}

/// One table's region of its arena's buffer, and its shape.
#[derive(Clone)]
pub struct ArenaColumn<E: CubeElement> {
    buffer: Buffer<E>,
    shape: Vec<usize>,
    placement: ArenaPlacement,
}

impl<E: CubeElement> ArenaColumn<E> {
    /// The region's elements, sharing the arena's memory.
    pub fn buffer(&self) -> &Buffer<E> {
        &self.buffer
    }

    /// The table's shape, first axis first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn placement(&self) -> &ArenaPlacement {
        &self.placement
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// `out[i] = value` over a launch's units.
#[cube(launch_unchecked)]
fn fill_elements<E: CubePrimitive + LaunchArg<RuntimeArg = E> + Send + Sync>(out: &mut [E], value: E) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = value;
    }
}

/// Fills `len` elements of `handle` from element `at` with `value`'s bytes.
fn fill_region<E: CubeElement + CubePrimitive + LaunchArg<RuntimeArg = E> + Send + Sync>(
    client: &Client,
    handle: &Handle,
    at: usize,
    len: usize,
    value: &[u8],
) {
    if len == 0 {
        return;
    }
    let value = E::from_bytes(value)[0];
    let lanes = 256usize;
    let cubes = len.div_ceil(lanes);
    let (x, y) = if cubes <= 65_535 { (cubes, 1) } else { (65_535, cubes.div_ceil(65_535)) };
    let region = Buffer::<E>::from_handle(handle.clone(), at + len).slice(at..at + len);
    // SAFETY: the kernel guards its index by the region's length.
    unsafe {
        fill_elements::launch_unchecked::<E>(
            client,
            CubeCount::Static(x as u32, y as u32, 1),
            CubeDim::new_1d(lanes as u32),
            (&region).into(),
            value,
        );
    }
}

#[doc(hidden)]
pub type ArenaMarker<T> = PhantomData<T>;
