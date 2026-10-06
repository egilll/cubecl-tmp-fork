use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn invalid(storage: &mut Storage<f32, ReadOnly>) {
    storage.store(0usize, 1.0);
}

fn main() {}
