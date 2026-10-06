use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
struct Distance(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
struct Time(f32);

#[repr(transparent)]
#[derive(CubeType, DeviceRepr)]
struct Speed(f32);

#[cube(inline)]
fn primitive(distance: f32, time: f32) -> f32 {
    distance / time
}

#[cube(inline)]
fn branded(distance: Distance, time: Time) -> Speed {
    Speed(distance.0 / time.0)
}

fn inputs(scope: &Scope) -> (NativeExpand<f32>, NativeExpand<f32>) {
    let ty = f32::__expand_as_type(scope);
    (
        scope.create_local_mut(ty, None).into(),
        scope.create_local_mut(ty, None).into(),
    )
}

fn scope() -> Scope {
    Scope::root(KernelSettings::new(
        CubeDim::new_1d(1).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

#[test]
fn transparent_arithmetic_has_identical_ir() {
    let raw = scope();
    let (distance, time) = inputs(&raw);
    let _ = primitive::expand(&raw, distance, time);

    let typed = scope();
    let (distance, time) = inputs(&typed);
    let _ = branded::expand(
        &typed,
        Distance::expand_from_repr(distance),
        Time::expand_from_repr(time),
    );
    assert_eq!(raw.to_string(), typed.to_string());
}

#[test]
fn forwarding_does_not_emit_operations() {
    let scope = scope();
    let (value, _) = inputs(&scope);
    let before = scope.to_string();
    let wrapped = Distance::expand_from_repr(value);
    let unwrapped = Distance::expand_into_repr(wrapped);
    assert_eq!(unwrapped.expand, value.expand);
    assert_eq!(before, scope.to_string());
}

#[cube(inline(never))]
fn outlined_primitive(distance: f32, time: f32) -> f32 {
    distance / time
}

#[cube(inline(never))]
fn outlined_branded(distance: Distance, time: Time) -> Speed {
    Speed(distance.0 / time.0)
}

#[test]
fn outlined_quantities_have_identical_ir() {
    let raw = scope();
    let (distance, time) = inputs(&raw);
    let _ = outlined_primitive::expand(&raw, distance, time);

    let typed = scope();
    let (distance, time) = inputs(&typed);
    let _ = outlined_branded::expand(
        &typed,
        Distance::expand_from_repr(distance),
        Time::expand_from_repr(time),
    );
    assert_eq!(raw.state().device_fns.len(), 1);
    assert_eq!(typed.state().device_fns.len(), 1);
    assert_eq!(
        raw.to_string(),
        typed
            .to_string()
            .replace("outlined_branded", "outlined_primitive"),
    );
}

#[cube(inline)]
fn raw_storage(input: &[f32], time: &[f32], output: &mut [f32]) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output[index] = input[index] / time[index];
    }
}

#[cube(inline)]
fn branded_storage(
    input: &Storage<Distance, ReadOnly>,
    time: &Storage<Time, ReadOnly>,
    output: &mut Storage<Speed>,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output.store(index, Speed(input.load(index).0 / time.load(index).0));
    }
}

fn builder() -> KernelBuilder {
    KernelBuilder::new(KernelSettings::new(
        CubeDim::new_1d(32).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    ))
}

#[test]
fn typed_storage_has_identical_bindings_and_ir() {
    let arg = BufferCompilationArg { inplace: None };
    let mut raw = builder();
    let input = <[f32]>::expand(&arg, &mut raw);
    let time = <[f32]>::expand(&arg, &mut raw);
    let mut output = <[f32]>::expand(&arg, &mut raw);
    raw_storage::expand(&raw.scope, &input, &time, &mut output);

    let mut typed = builder();
    let input = <Storage<Distance, ReadOnly>>::expand(&arg, &mut typed);
    let time = <Storage<Time, ReadOnly>>::expand(&arg, &mut typed);
    let mut output = <Storage<Speed>>::expand(&arg, &mut typed);
    branded_storage::expand(&typed.scope, &input, &time, &mut output);
    assert_eq!(raw.scope.to_string(), typed.scope.to_string());
}

struct Room;
struct World;

#[derive(CubeType)]
struct Position<F: 'static>(f32, #[cube(comptime)] core::marker::PhantomData<F>);

#[cube(inline(never))]
fn shift<F: 'static>(point: Position<F>) -> Position<F> {
    Position::<F>(point.0 + 1.0, comptime! { core::marker::PhantomData })
}

#[test]
fn phantom_frames_have_no_slots_and_distinguish_specializations() {
    use core::hash::Hasher;
    use cubecl::frontend::call::CallArg;

    let scope = scope();
    let (value, _) = inputs(&scope);
    let room = PositionExpand::<Room>(value, core::marker::PhantomData);
    let world = PositionExpand::<World>(value, core::marker::PhantomData);
    let mut slots = Vec::new();
    assert!(room.call_slots(&scope, &mut slots));
    assert_eq!(slots.len(), 1);
    assert_eq!(room.call_returns(&scope).unwrap().len(), 1);

    let mut room_key = std::collections::hash_map::DefaultHasher::new();
    let mut world_key = std::collections::hash_map::DefaultHasher::new();
    assert!(room.call_key(&mut room_key));
    assert!(world.call_key(&mut world_key));
    assert_ne!(room_key.finish(), world_key.finish());
    let _ = shift::expand::<Room>(&scope, room);
    let _ = shift::expand::<World>(&scope, world);
    assert_eq!(scope.state().device_fns.len(), 2);
}
