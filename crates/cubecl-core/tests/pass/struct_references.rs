//! A `CubeType` struct is borrowed, reborrowed and copied as in Rust.
use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Memory {
    stability: f32,
    difficulty: f32,
}

#[cube]
fn read(memory: &Memory) -> f32 {
    memory.stability + memory.difficulty
}

#[cube]
trait Law {
    fn apply(&self, memory: &mut Memory) -> f32;
}

#[derive(CubeType)]
struct Decay;

#[cube]
impl Law for Decay {
    fn apply(&self, memory: &mut Memory) -> f32 {
        let total = read(memory);
        let reborrowed = read(&*memory);
        let mut copy = *memory;
        copy.stability = 0.0;
        total + reborrowed + copy.difficulty
    }
}

#[cube]
fn sum(values: &[f32]) -> f32 {
    values[0] + values[1]
}

#[cube(launch)]
fn kernel(values: &mut [f32], output: &mut [f32]) {
    output[0] = sum(values);
}

fn main() {}
