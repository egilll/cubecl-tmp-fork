use cubecl_core::prelude::*;
use cubecl_runtime::runtime::Runtime;

use crate::matmul::{View, matmul};

fn values(count: usize, seed: u32) -> Vec<f32> {
    (0..count as u32)
        .map(|i| ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

/// `C ← α A B + β C` on the host, every matrix through its view.
fn reference(
    a: &[f32],
    av: View,
    b: &[f32],
    bv: View,
    c: &mut [f32],
    cv: View,
    alpha: f32,
    beta: f32,
    batches: usize,
) {
    for batch in 0..batches {
        for i in 0..cv.rows {
            for j in 0..cv.columns {
                let mut sum = 0.0f64;
                for k in 0..av.columns {
                    let x = a[av.offset
                        + batch * av.batch_stride
                        + i * av.row_stride
                        + k * av.column_stride];
                    let y = b[bv.offset
                        + batch * bv.batch_stride
                        + k * bv.row_stride
                        + j * bv.column_stride];
                    sum += f64::from(x) * f64::from(y);
                }
                let at =
                    cv.offset + batch * cv.batch_stride + i * cv.row_stride + j * cv.column_stride;
                c[at] = alpha * sum as f32 + beta * c[at];
            }
        }
    }
}

/// Square, ragged, transposed, batched and long-inner (split) products
/// agree with the host.
pub fn test_matmul<R: Runtime>(client: Client) {
    // (rows, columns, inner, batches, transposed A)
    let cases = [
        (1, 1, 1, 1, false),
        (16, 16, 16, 1, false),
        (37, 21, 45, 1, false),
        (33, 17, 9, 3, false),
        (24, 24, 20_000, 1, true),
        (5, 64, 300, 2, true),
    ];
    for (rows, columns, inner, batches, transposed) in cases {
        let av = if transposed {
            View::row_major(inner, rows).transposed()
        } else {
            View::row_major(rows, inner)
        };
        let av = View {
            batch_stride: rows * inner,
            ..av
        };
        let bv = View::row_major(inner, columns);
        let cv = View::row_major(rows, columns);
        let a = values(rows * inner * batches, 1);
        let b = values(inner * columns * batches, 2);
        let mut c = values(rows * columns * batches, 3);
        let (alpha, beta) = (0.75f32, -0.5f32);
        let c_buffer = Buffer::create(&client, &c);
        matmul(
            &client,
            (&Buffer::create(&client, &a), av),
            (&Buffer::create(&client, &b), bv),
            (&c_buffer, cv),
            alpha,
            beta,
            batches,
        );
        reference(&a, av, &b, bv, &mut c, cv, alpha, beta, batches);
        let got = c_buffer.read(&client).unwrap();
        let scale = (inner as f32).sqrt();
        for (index, (got, want)) in got.iter().zip(&c).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5 * scale * want.abs().max(1.0) + 1e-5 * scale,
                "{rows}×{columns}×{inner}×{batches}: {index}: {got} vs {want}"
            );
        }
    }
}

#[macro_export]
macro_rules! testgen_matmul {
    () => {
        mod matmul {
            use super::*;

            #[$crate::tests::test_log::test]
            fn test_matmul() {
                let client = TestRuntime::client(&Default::default());
                cubecl_algorithms::tests::matmul::test_matmul::<TestRuntime>(client);
            }
        }
    };
}
