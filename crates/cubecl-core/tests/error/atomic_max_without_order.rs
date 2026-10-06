use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Ordered on the host, but not as its representation.
#[repr(transparent)]
#[derive(CubeType, DeviceRepr, PartialEq)]
#[device_repr(copy)]
struct Reversed(u32);

impl PartialOrd for Reversed {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

#[cube]
fn largest(values: &AtomicStorage<Reversed>) {
    values.fetch_max(0, Reversed(1));
}

fn main() {}
