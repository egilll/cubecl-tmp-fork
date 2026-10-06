use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
#[device_repr(copy)]
struct Label(u32);

#[cube]
fn count(labels: &AtomicStorage<Label>) {
    labels.fetch_add(0, Label(1));
}

fn main() {}
