//! Fixed-index local arrays are scalarized before target declarations replace them.

use cubecl_core as cubecl;
use cubecl_core::{Compiler, prelude::*};
use cubecl_cpp::{shared::CppCompiler, target};

#[cube(inline)]
fn accumulate(values: &mut [f32]) {
    let mut sums = Array::<f32>::new(12usize);
    #[unroll]
    for i in 0..12usize {
        sums[i] = 0.0;
    }
    for j in 0..values.len() {
        #[unroll]
        for a in 0..3usize {
            #[unroll]
            for f in 0..4usize {
                sums[a * 4 + f] += values[j] * (a + f) as f32;
            }
        }
    }
    #[unroll]
    for i in 0..12usize {
        values[ABSOLUTE_POS + i] = sums[i];
    }
}

fn check<C: Compiler + Default>()
where
    C::CompilationOptions: Default,
{
    let mut builder = KernelBuilder::new(KernelSettings::new(
        CubeDim::new_1d(1).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    ));
    let mut values = <[f32]>::expand(&BufferCompilationArg { inplace: None }, &mut builder);
    accumulate::expand(&builder.scope, &mut values);
    let source = C::default()
        .compile(builder.build(), &Default::default())
        .unwrap()
        .to_string();
    assert!(!source.contains("array<float, 12>"), "{source}");
    assert!(!source.contains("[12]"), "{source}");
}

#[test]
fn cuda_scalarizes_fixed_index_arrays() {
    check::<CppCompiler<target::Cuda>>();
}

#[test]
fn hip_scalarizes_fixed_index_arrays() {
    check::<CppCompiler<target::Hip>>();
}

#[cfg(feature = "metal")]
#[test]
fn metal_scalarizes_fixed_index_arrays() {
    check::<cubecl_cpp::MslCompiler>();
}
