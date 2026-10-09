use crate as cubecl;
use crate::prelude::*;

/// A value `?` unwraps in kernels. `value?` in a statement of a function
/// body continues with [`output`](Self::output) when the value is
/// [`present`](Self::present), and otherwise makes the function return its
/// own type's [`absent`](Self::absent) value.
#[cube]
pub trait Fallible: CubeType + Sized {
    type Output: CubeType;

    /// Whether the value is present, and its output, meaningful only when
    /// it is.
    fn split(self) -> (bool, Self::Output);

    fn absent() -> Self;
}
