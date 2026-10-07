//! Cube-wide reductions over typed values, subgroup first.
//!
//! A butterfly of shuffles combines a subgroup's values with no shared
//! memory and no barrier; then one value per subgroup crosses through
//! typed shared storage, with one barrier pair. Every unit receives the
//! result, and values keep their brands throughout. Every unit of the cube
//! must call a collective, in uniform control flow.

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// The scratch a cube collective needs: one entry per subgroup of a cube of
/// `units` units, at the narrowest subgroup a backend reports (4 units).
pub const fn subgroups(units: usize) -> usize {
    units.div_ceil(4)
}

/// A value collectives can move: a brand over a primitive, copied freely.
pub trait Collective: DeviceRepr + CubeType<ExpandType: Clone + Copy> + Copy + Send + Sync {}
impl<Q: DeviceRepr + CubeType<ExpandType: Clone + Copy> + Copy + Send + Sync> Collective for Q {}

/// The sum of every unit's `value`.
#[cube]
pub fn cube_sum<Q: Collective + DeviceRepr<Repr = f32>>(value: Q, scratch: &mut SharedStorage<Q>) -> Q {
    let partial = plane_sum(Q::into_repr(value));
    if UNIT_POS_PLANE == 0 {
        scratch.store((UNIT_POS / PLANE_DIM) as usize, Q::from_repr(partial));
    }
    sync_cube();
    let mut total = 0.0f32;
    for plane in 0..CUBE_DIM.div_ceil(PLANE_DIM) {
        total += Q::into_repr(scratch.load(plane as usize));
    }
    sync_cube();
    Q::from_repr(total)
}

/// One unit's offer to an argmax: its rank and what it names.
#[derive(CubeType, CubeTypeMut, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct Best<R: Collective + DeviceRepr<Repr = f32>, I: Collective + DeviceRepr<Repr = u32>> {
    pub rank: R,
    pub index: I,
}

/// Whether `(rank, index)` beats `(best, at)`: higher, ties to the lower index.
#[cube]
fn beats(rank: f32, index: u32, best: f32, at: u32) -> bool {
    rank > best || (rank == best && index < at)
}

/// The best offer of the cube: the highest rank, ties to the lower index,
/// so how units are batched never changes the winner.
#[cube]
pub fn cube_argmax<R: Collective + DeviceRepr<Repr = f32>, I: Collective + DeviceRepr<Repr = u32>>(
    offer: Best<R, I>,
    ranks: &mut SharedStorage<R>,
    indices: &mut SharedStorage<I>,
) -> Best<R, I> {
    let mut rank = R::into_repr(offer.rank);
    let mut index = I::into_repr(offer.index);
    let mut mask = 1u32;
    while mask < PLANE_DIM {
        let other = plane_shuffle_xor(rank, mask);
        let named = plane_shuffle_xor(index, mask);
        let take = beats(other, named, rank, index);
        rank = select(take, other, rank);
        index = select(take, named, index);
        mask *= 2;
    }
    if UNIT_POS_PLANE == 0 {
        let plane = (UNIT_POS / PLANE_DIM) as usize;
        ranks.store(plane, R::from_repr(rank));
        indices.store(plane, I::from_repr(index));
    }
    sync_cube();
    let mut best = R::into_repr(ranks.load(0));
    let mut at = I::into_repr(indices.load(0));
    for plane in 1..CUBE_DIM.div_ceil(PLANE_DIM) {
        let other = R::into_repr(ranks.load(plane as usize));
        let named = I::into_repr(indices.load(plane as usize));
        let take = beats(other, named, best, at);
        best = select(take, other, best);
        at = select(take, named, at);
    }
    sync_cube();
    Best::<R, I> { rank: R::from_repr(best), index: I::from_repr(at) }
}

/// The OR of every unit's `bits` within its subgroup, by shuffles.
#[cube]
pub fn plane_or(bits: u32) -> u32 {
    let mut bits = bits;
    let mut mask = 1u32;
    while mask < PLANE_DIM {
        bits |= plane_shuffle_xor(bits, mask);
        mask *= 2;
    }
    bits
}
