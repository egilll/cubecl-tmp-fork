//! A kernel is launched from code that forbids `unsafe`, through typed buffers.
#![forbid(unsafe_code)]

use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_core::compute::Buffer;

#[cube(launch)]
fn scale(input: &[f32], output: &mut [f32], factor: f32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * factor;
    }
}

#[allow(dead_code)]
fn run(client: &Client) -> Vec<f32> {
    let input = Buffer::create(client, &[1.0f32, 2.0, 3.0]);
    let output = Buffer::<f32>::empty(client, 3);
    scale::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(3),
        (&input).into(),
        (&output).into(),
        2.0,
    );
    output.read(client).unwrap()
}

fn main() {}
