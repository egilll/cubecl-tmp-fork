//! Batched complex FFT of power-of-two length: radix-2 Stockham, one launch
//! per pass, ping-ponging between two buffers so no pass reorders in place
//! and no bit reversal is needed.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

const UNITS: u32 = 256;

/// One radix-2 pass with butterfly span `span` over complex values stored
/// interleaved (`re, im`), `len` per transform. Unit `t` computes butterfly
/// `t % (len / 2)` of transform `t / (len / 2)`.
#[cube(launch)]
fn stockham_pass(input: &[f32], output: &mut [f32], len: u32, span: u32, sign: f32, scale: f32) {
    let half = (len / 2) as usize;
    let t = ABSOLUTE_POS;
    if t * 2 < input.len() / 2 {
        let base = (t / half) * len as usize;
        let j = t % half;
        let span = span as usize;
        let k = j % span;
        let angle = sign * 6.283_185_5 * k as f32 / (2 * span) as f32;
        let (wr, wi) = (f32::cos(angle), f32::sin(angle));
        let a = (base + j) * 2;
        let b = (base + j + half) * 2;
        let (ar, ai) = (input[a], input[a + 1]);
        let (br, bi) = (input[b], input[b + 1]);
        let (cr, ci) = (br * wr - bi * wi, br * wi + bi * wr);
        let d = (base + (j / span) * span * 2 + k) * 2;
        let e = d + span * 2;
        output[d] = (ar + cr) * scale;
        output[d + 1] = (ai + ci) * scale;
        output[e] = (ar - cr) * scale;
        output[e + 1] = (ai - ci) * scale;
    }
}

/// The discrete Fourier transforms of the length-`len` complex sequences in
/// `input`, stored back to back with interleaved `re, im` parts (so
/// `input.len()` is `2 * len * batch`). Forward is `X[k] = Σ x[j] e^(-2πijk/len)`;
/// `inverse` flips the exponent's sign and divides by `len`, so a forward
/// then an inverse transform round-trips. `len` is a power of two.
pub fn fft(client: &Client, input: &Buffer<f32>, len: usize, inverse: bool) -> Buffer<f32> {
    assert!(len.is_power_of_two(), "the length is a power of two");
    assert_eq!(
        input.len() % (2 * len),
        0,
        "whole transforms of interleaved complex values"
    );
    let output = Buffer::<f32>::empty(client, input.len());
    let other = Buffer::<f32>::empty(client, input.len());
    if input.is_empty() {
        return output;
    }
    let passes = len.trailing_zeros();
    let butterflies = input.len() / 4;
    let dim = CubeDim::new_1d(UNITS);
    // Land the last pass in `output`: start there when the count is odd.
    let (mut to, mut spare) = match passes % 2 {
        1 => (&output, &other),
        _ => (&other, &output),
    };
    let mut from = input;
    for pass in 0..passes {
        let scale = match inverse && pass + 1 == passes {
            true => 1.0 / len as f32,
            false => 1.0,
        };
        stockham_pass::launch(
            client,
            cubecl_core::calculate_cube_count_elemwise(client, butterflies.max(1), dim),
            dim,
            from.into(),
            to.into(),
            len as u32,
            1 << pass,
            if inverse { 1.0 } else { -1.0 },
            scale,
        );
        from = to;
        core::mem::swap(&mut to, &mut spare);
    }
    if passes == 0 {
        // A length-one transform is the identity.
        return input.clone();
    }
    output
}
