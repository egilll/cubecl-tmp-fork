use core::ops::{Add, AddAssign, Mul, Neg, Sub};

use crate::prelude::*;
use crate::{self as cubecl};
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

#[derive(CubeType, CubeTypeMut, Clone, Copy, Debug, PartialEq)]
#[expand(derive(Clone, Copy))]
pub struct Pair {
    re: f32,
    im: f32,
}

#[cube]
impl Add for Pair {
    type Output = Pair;
    fn add(self, rhs: Pair) -> Pair {
        Pair {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

#[cube]
impl Sub for Pair {
    type Output = Pair;
    fn sub(self, rhs: Pair) -> Pair {
        Pair {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

#[cube]
impl Mul<f32> for Pair {
    type Output = Pair;
    fn mul(self, rhs: f32) -> Pair {
        Pair {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

#[cube]
impl Neg for Pair {
    type Output = Pair;
    fn neg(self) -> Pair {
        Pair {
            re: -self.re,
            im: -self.im,
        }
    }
}

#[cube]
impl AddAssign for Pair {
    fn add_assign(&mut self, rhs: Pair) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

/// Written once: traced on the device, and plain Rust on the host.
#[cube]
fn formula(a: Pair, b: Pair, s: f32) -> Pair {
    let mut c = (a + b) * s - -a;
    c += b;
    c
}

#[cube(launch)]
fn kernel_user_ops(input: &[f32], output: &mut [f32]) {
    let i = ABSOLUTE_POS;
    if i < input.len() / 4 {
        let a = Pair {
            re: input[4 * i],
            im: input[4 * i + 1],
        };
        let b = Pair {
            re: input[4 * i + 2],
            im: input[4 * i + 3],
        };
        let c = formula(a, b, 0.5);
        output[2 * i] = c.re;
        output[2 * i + 1] = c.im;
    }
}

/// Operators implemented on a user type trace into the user's methods, and
/// compute what the same formula computes on the host.
pub fn test_user_operators<R: Runtime>(client: Client) {
    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.37 - 7.0).collect();
    let input_buffer = crate::compute::Buffer::create(&client, &input);
    let output = crate::compute::Buffer::<f32>::empty(&client, 32);
    kernel_user_ops::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(16),
        (&input_buffer).into(),
        (&output).into(),
    );
    let output = output.read(&client).unwrap();
    for i in 0..16 {
        let a = Pair {
            re: input[4 * i],
            im: input[4 * i + 1],
        };
        let b = Pair {
            re: input[4 * i + 2],
            im: input[4 * i + 3],
        };
        let c = formula(a, b, 0.5);
        for (actual, expected) in [(output[2 * i], c.re), (output[2 * i + 1], c.im)] {
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "element {i}: {actual} vs {expected}"
            );
        }
    }
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_user_ops {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_user_operators() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::user_ops::test_user_operators::<TestRuntime>(client);
        }
    };
}
