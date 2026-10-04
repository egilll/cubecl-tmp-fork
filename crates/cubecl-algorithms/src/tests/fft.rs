use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::fft::fft;

/// The direct `O(n²)` DFT, in `f64`.
fn reference(input: &[f32], len: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.len()];
    for (b, x) in input.chunks(2 * len).enumerate() {
        for k in 0..len {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for j in 0..len {
                let angle = -2.0 * core::f64::consts::PI * (j * k % len) as f64 / len as f64;
                let (xr, xi) = (x[2 * j] as f64, x[2 * j + 1] as f64);
                re += xr * angle.cos() - xi * angle.sin();
                im += xr * angle.sin() + xi * angle.cos();
            }
            out[2 * (b * len + k)] = re;
            out[2 * (b * len + k) + 1] = im;
        }
    }
    out
}

pub fn test_fft<R: Runtime>(client: Client) {
    for (len, batch) in [(1usize, 3usize), (2, 1), (8, 5), (64, 2), (1024, 3)] {
        let input: Vec<f32> = (0..2 * len * batch)
            .map(|i| ((i as u32).wrapping_mul(2654435761) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let buffer = Buffer::create(&client, &input);
        let forward = fft(&client, &buffer, len, false);
        let got = forward.read(&client).unwrap();
        let expected = reference(&input, len);
        let tolerance = 1e-5 * (len as f64).sqrt() * (len as f64).log2().max(1.0) * 4.0;
        for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (*g as f64 - e).abs() <= tolerance * (1.0 + e.abs()),
                "len {len}, index {i}: {g} vs {e}"
            );
        }
        // Parseval: energy is preserved up to the factor `len`.
        let energy = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>();
        let ratio = energy(&got) / (energy(&input) * len as f64);
        assert!(
            (ratio - 1.0).abs() < 1e-4,
            "len {len}: Parseval ratio {ratio}"
        );
        // Round trip.
        let back = fft(&client, &forward, len, true).read(&client).unwrap();
        for (i, (b, x)) in back.iter().zip(&input).enumerate() {
            assert!(
                (b - x).abs() < 1e-5 * len as f32 + 1e-6,
                "len {len}, index {i}: {b} vs {x}"
            );
        }
    }
}

#[macro_export]
macro_rules! testgen_fft {
    () => {
        mod fft {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_fft() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::fft::test_fft::<TestRuntime>(client);
            }
        }
    };
}
