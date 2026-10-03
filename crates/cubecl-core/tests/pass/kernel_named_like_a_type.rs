//! A kernel can be named after a type its code uses: the items generated for it, named after the
//! kernel, do not shadow it.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch)]
fn shared(output: &mut [f32]) {
    let mut smem = Shared::<[f32]>::new_slice(1usize);
    smem[0] = 1.0;
    output[0] = smem[0];
}

#[cube(launch)]
fn array<F: Float>(output: &mut [F]) {
    let mut local = Array::<F>::new(1usize);
    local[0] = F::new(1.0f32);
    output[0] = local[0];
}

fn main() {
    let _ = shared::launch;
    let _ = array::launch::<f32>;
}
