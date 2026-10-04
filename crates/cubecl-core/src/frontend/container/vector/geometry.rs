use cubecl_macros::{cube, intrinsic};

use crate::{self as cubecl, prelude::*};

use super::Vector;
type VectorExpand<E, N> = NativeExpand<Vector<E, N>>;

#[cube]
impl<F: Float, N: Size> Vector<F, N> {
    /// The cross product of two 3-vectors.
    ///
    /// # Panics
    ///
    /// While tracing, when the vectors don't have exactly three lanes.
    pub fn cross(&self, other: &Self) -> Self {
        require_three_lanes::<N>();
        let a = *self;
        let b = *other;
        let mut out = a;
        out.insert(
            0usize,
            a.extract(1usize) * b.extract(2usize) - a.extract(2usize) * b.extract(1usize),
        );
        out.insert(
            1usize,
            a.extract(2usize) * b.extract(0usize) - a.extract(0usize) * b.extract(2usize),
        );
        out.insert(
            2usize,
            a.extract(0usize) * b.extract(1usize) - a.extract(1usize) * b.extract(0usize),
        );
        out
    }
}

// `N` is read inside the intrinsic, which clippy doesn't see.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
fn require_three_lanes<N: Size>() {
    intrinsic!(|scope| {
        assert_eq!(
            N::__expand_value(scope),
            3,
            "`cross` takes vectors of exactly three lanes"
        );
    })
}
