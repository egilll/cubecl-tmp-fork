//! An untyped literal takes its type from the other operand, as in Rust, on either side.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube(launch)]
fn kernel(output: &mut [f32], flags: &mut [u32], u: f32, decay: f32, truly: f32, bits: u32) {
    let correct = u > 0.5;
    output[0] = f32::sqrt(-2.0 * f32::ln(u));
    output[1] = select(correct, 1.0, decay);
    output[2] = select(correct, 1.0, 0.0) - truly;
    output[3] = f32::powf(0.9, -1.0 / decay);
    output[4] = (f32::cast_from(bits >> 8) + 0.5) * (1.0 / 16_777_216.0);
    if 0.5 < u {
        flags[0] = 1 << bits;
    }
}

fn main() {
    let _ = kernel::launch;
}
