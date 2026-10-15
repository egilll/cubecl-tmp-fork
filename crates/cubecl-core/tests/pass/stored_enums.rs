use cubecl::prelude::*;
use cubecl_core as cubecl;

/// A fieldless runtime enum stored as its discriminant.
#[derive(CubeType, CubeLaunch, CubeTypeMut, IntoRuntime, DeviceRepr, Clone, Copy, PartialEq, Eq, Debug)]
#[cube(runtime_variants)]
#[device_repr(copy, eq)]
enum Family {
    Index,
    Threshold,
    Fill,
}

#[cube]
fn rate(family: Family, x: f32) -> f32 {
    match family {
        Family::Index => x,
        Family::Threshold => 2.0 * x,
        Family::Fill => 3.0,
    }
}

/// Unit variants are values, so an `if` can return one.
#[cube]
fn pick(flag: bool) -> Family {
    if flag { Family::Index } else { Family::Fill }
}

#[cube(launch)]
fn kernel(families: &mut Storage<Family>, out: &mut [f32]) {
    let family = families.load(0usize);
    let picked = pick(family == Family::Threshold);
    families.store(1usize, picked);
    out[0] = rate(picked, 1.5);
}

/// An enum keys a table's axis by its discriminant.
#[derive(CubeType, CubeTypeMut, IntoRuntime, DeviceRepr, Clone, Copy, PartialEq, Eq, Debug)]
#[cube(runtime_variants)]
#[device_repr(copy, eq, key)]
enum Balance {
    Felt,
    Active,
}

#[cube(launch)]
fn keyed(table: &mut Storage<f32, ReadWrite, Balance>) {
    table.store(Balance::Active, 2.0);
}

/// An enum passed to a kernel as a scalar.
#[derive(CubeType, CubeTypeMut, IntoRuntime, DeviceRepr, Clone, Copy, PartialEq, Eq, Debug)]
#[cube(runtime_variants)]
#[device_repr(copy, eq, launch)]
enum Mode {
    Off,
    On,
}

#[cube(launch)]
fn moded(mode: Mode, out: &mut [f32]) {
    if mode == Mode::On {
        out[0] = 1.0;
    }
}

fn main() {
    assert_eq!(Family::from_repr(Family::Fill.into_repr()), Family::Fill);
}
