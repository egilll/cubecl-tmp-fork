//! Batched matrix products of strided views: `C ← α A B + β C`.
//!
//! Each view names its offset, shape and row and column strides in
//! elements, so a transpose is a view with its strides swapped and a batch
//! is a further stride, with no copies. Cubes compute square tiles of `C`
//! from tiles of `A` and `B` staged in shared memory. A product whose inner
//! dimension is long against its output, such as the Gram matrix `Xᵀ Y` of
//! two tall, thin blocks, is split along it: each cube sums one stretch,
//! and a second pass adds the stretches, so no cube waits on another.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

/// Rows and columns of a cube's tile of `C`, and the stretch of the inner
/// dimension staged at once.
const TILE: usize = 16;

/// A matrix in a buffer: where it starts, its shape, its strides between
/// rows and between columns, and between the matrices of a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct View {
    pub offset: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: usize,
    pub column_stride: usize,
    pub batch_stride: usize,
}

impl View {
    /// A dense row-major matrix.
    pub fn row_major(rows: usize, columns: usize) -> Self {
        Self {
            offset: 0,
            rows,
            columns,
            row_stride: columns,
            column_stride: 1,
            batch_stride: rows * columns,
        }
    }

    /// The same storage read as its transpose.
    pub fn transposed(self) -> Self {
        Self {
            rows: self.columns,
            columns: self.rows,
            row_stride: self.column_stride,
            column_stride: self.row_stride,
            ..self
        }
    }

    /// The last element any matrix of `batch` reads, plus one.
    fn extent(&self, batch: usize) -> usize {
        if self.rows == 0 || self.columns == 0 || batch == 0 {
            return 0;
        }
        self.offset
            + (batch - 1) * self.batch_stride
            + (self.rows - 1) * self.row_stride
            + (self.columns - 1) * self.column_stride
            + 1
    }
}

/// A view as kernels read it.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct Strides {
    offset: u32,
    row: u32,
    column: u32,
    batch: u32,
}

impl View {
    fn strides(&self) -> StridesLaunch {
        let narrow = |value: usize| u32::try_from(value).expect("matrix views fit 32-bit indices");
        StridesLaunch::new(
            narrow(self.offset),
            narrow(self.row_stride),
            narrow(self.column_stride),
            narrow(self.batch_stride),
        )
    }
}

#[cube]
fn at(strides: &Strides, batch: usize, row: usize, column: usize) -> usize {
    strides.offset as usize
        + batch * strides.batch as usize
        + row * strides.row as usize
        + column * strides.column as usize
}

/// One cube per tile of `C` and stretch of the inner dimension: the
/// stretch's share of `A B` for the tile, into `partials` (one matrix per
/// batch and stretch, row-major).
#[cube(launch)]
#[allow(clippy::too_many_arguments, reason = "three views and their shape")]
fn tiles<F: Float>(
    a: &[F],
    a_strides: &Strides,
    b: &[F],
    b_strides: &Strides,
    partials: &mut [F],
    shape: &Shape,
) {
    let mut a_tile = Shared::<[F]>::new_slice(TILE * TILE);
    let mut b_tile = Shared::<[F]>::new_slice(TILE * TILE);
    let (rows, columns, inner) = (
        shape.rows as usize,
        shape.columns as usize,
        shape.inner as usize,
    );
    let (stretch, splits) = (shape.stretch as usize, shape.splits as usize);
    let (local_row, local_column) = (UNIT_POS_Y, UNIT_POS_X);
    let row = CUBE_POS_Y * TILE + local_row;
    let column = CUBE_POS_X * TILE + local_column;
    let batch = CUBE_POS_Z / splits;
    let split = CUBE_POS_Z % splits;
    let first = split * stretch;
    let last = usize::min(first + stretch, inner);
    let mut sum = F::from_int(0);
    let mut start = first;
    while start < last {
        // Stage A[row, start + local_column] and B[start + local_row, column].
        let a_inner = start + local_column;
        let b_inner = start + local_row;
        let mut a_value = F::from_int(0);
        if row < rows && a_inner < last {
            a_value = a[at(a_strides, batch, row, a_inner)];
        }
        let mut b_value = F::from_int(0);
        if b_inner < last && column < columns {
            b_value = b[at(b_strides, batch, b_inner, column)];
        }
        a_tile[local_row * TILE + local_column] = a_value;
        b_tile[local_row * TILE + local_column] = b_value;
        sync_cube();
        #[unroll]
        for k in 0..TILE {
            sum += a_tile[local_row * TILE + k] * b_tile[k * TILE + local_column];
        }
        sync_cube();
        start += TILE;
    }
    if row < rows && column < columns {
        partials[(CUBE_POS_Z * rows + row) * columns + column] = sum;
    }
}

/// A product's shape: `C` is `rows × columns`, the inner dimension `inner`,
/// split into `splits` stretches of `stretch`.
#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct Shape {
    rows: u32,
    columns: u32,
    inner: u32,
    stretch: u32,
    splits: u32,
    batches: u32,
}

/// One unit per element of `C`: the stretches' sum, scaled by `scales[0]`
/// and added to `scales[1]` times what `C` held (nothing when that is
/// zero).
#[cube(launch)]
fn finish<F: Float>(partials: &[F], c: &mut [F], c_strides: &Strides, shape: &Shape, scales: &[F]) {
    let (rows, columns) = (shape.rows as usize, shape.columns as usize);
    let (splits, batches) = (shape.splits as usize, shape.batches as usize);
    let index = ABSOLUTE_POS;
    if index >= rows * columns * batches {
        terminate!();
    }
    let batch = index / (rows * columns);
    let row = (index / columns) % rows;
    let column = index % columns;
    let mut sum = F::from_int(0);
    for split in 0..splits {
        sum += partials[((batch * splits + split) * rows + row) * columns + column];
    }
    let target = at(c_strides, batch, row, column);
    let (alpha, beta) = (scales[0], scales[1]);
    let mut previous = F::from_int(0);
    if beta != F::from_int(0) {
        previous = c[target];
    }
    c[target] = alpha * sum + beta * previous;
}

/// `C ← α A B + β C` for each of `batches` matrices of the views; `β = 0`
/// ignores what `C` held, NaNs included.
///
/// # Panics
///
/// When the shapes do not chain, or a view reaches past its buffer.
#[allow(clippy::too_many_arguments, reason = "three views and two scales")]
pub fn matmul<F: Float + CubeElement>(
    client: &Client,
    a: (&Buffer<F>, View),
    b: (&Buffer<F>, View),
    c: (&Buffer<F>, View),
    alpha: F,
    beta: F,
    batches: usize,
) {
    let ((a, a_view), (b, b_view), (c, c_view)) = (a, b, c);
    assert_eq!(a_view.columns, b_view.rows, "A's columns are B's rows");
    assert_eq!(a_view.rows, c_view.rows, "A's rows are C's rows");
    assert_eq!(
        b_view.columns, c_view.columns,
        "B's columns are C's columns"
    );
    assert!(
        a_view.extent(batches) <= a.len(),
        "A's view within its buffer"
    );
    assert!(
        b_view.extent(batches) <= b.len(),
        "B's view within its buffer"
    );
    assert!(
        c_view.extent(batches) <= c.len(),
        "C's view within its buffer"
    );
    let (rows, columns, inner) = (c_view.rows, c_view.columns, a_view.columns);
    if rows == 0 || columns == 0 || batches == 0 {
        return;
    }
    let narrow = |value: usize| u32::try_from(value).expect("matrix shapes fit 32-bit indices");
    // Split a long inner dimension until the grid holds a few hundred
    // cubes, each stretch whole tiles.
    let output_tiles = rows.div_ceil(TILE) * columns.div_ceil(TILE) * batches;
    let splits = (512 / output_tiles.max(1))
        .clamp(1, inner.div_ceil(TILE).max(1))
        .min(256);
    let stretch = inner.div_ceil(splits).div_ceil(TILE) * TILE;
    let splits = inner.div_ceil(stretch.max(1)).max(1);
    let partials = Buffer::<F>::empty(client, rows * columns * batches * splits);
    let shape = || {
        ShapeLaunch::new(
            narrow(rows),
            narrow(columns),
            narrow(inner),
            narrow(stretch),
            narrow(splits),
            narrow(batches),
        )
    };
    tiles::launch::<F>(
        client,
        CubeCount::Static(
            narrow(columns.div_ceil(TILE)),
            narrow(rows.div_ceil(TILE)),
            narrow(batches * splits),
        ),
        CubeDim::new_2d(TILE as u32, TILE as u32),
        a.into(),
        a_view.strides(),
        b.into(),
        b_view.strides(),
        (&partials).into(),
        shape(),
    );
    let count = rows * columns * batches;
    finish::launch::<F>(
        client,
        cubecl_core::calculate_cube_count_elemwise(client, count, CubeDim::new_1d(256)),
        CubeDim::new_1d(256),
        (&partials).into(),
        c.into(),
        c_view.strides(),
        shape(),
        (&Buffer::create(client, &[alpha, beta])).into(),
    );
}
