//! Kernel counterparts of `core::mem` functions; `#[cube]` maps calls to these.

use crate::prelude::*;
use crate::{self as cubecl};

/// Exchanges the values behind two mutable references, as `core::mem::swap`.
#[cube]
pub fn swap<T: CubePrimitive>(a: &mut T, b: &mut T) {
    let held = *a;
    *a = *b;
    *b = held;
}
