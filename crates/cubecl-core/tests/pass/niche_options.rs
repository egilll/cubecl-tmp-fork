use cubecl::prelude::*;
use cubecl_core as cubecl;

/// An ID whose `u32::MAX` names no card, so an absent one costs no extra word.
#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, IntoRuntime, DeviceRepr, Default, Debug)]
#[device_repr(copy, eq, key, launch, niche = u32::MAX)]
struct CardId(u32);

#[cube(launch)]
fn optional(ids: &mut Storage<Option<CardId>>, out: &mut [u32]) {
    let first = ids.load(0usize);
    out[0] = match first {
        Some(card) => card.0,
        None => 7u32,
    };
    out[1] = select(ids.load(1usize).is_some(), 1u32, 0u32);
    ids.store(2usize, first);
    ids.store(3usize, None);
    ids.store(4usize, Some(CardId(5)));
}

fn main() {
    assert_eq!(Option::<CardId>::from_repr(u32::MAX).map(|id| id.0), None);
    assert_eq!(Some(CardId(3)).into_repr(), 3);
}
