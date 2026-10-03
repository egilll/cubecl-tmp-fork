//! A kernel can name its variables after the ones generated code declares for itself.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[cube]
fn helper(scope: u32, value: u32) -> u32 {
    let _value = scope + value;
    _value * 2
}

#[cube(launch)]
fn kernel(output: &mut [u32], builder: u32, launcher: u32, settings: u32) {
    let scope = builder + launcher;
    let value = helper(scope, settings);
    match value {
        0 => output[0] = scope,
        _ => output[0] = value,
    }
}

fn main() {
    let _ = kernel::launch;
}
