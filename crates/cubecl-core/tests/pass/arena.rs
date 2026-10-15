use cubecl::prelude::*;
use cubecl_core as cubecl;

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, IntoRuntime, DeviceRepr, Default, Debug)]
#[device_repr(copy, eq, key, launch, niche = u32::MAX)]
struct CardId(u32);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, IntoRuntime, DeviceRepr, Default, Debug)]
#[device_repr(copy, eq, key, launch)]
struct Day(u32);

/// A record of mixed words.
#[derive(CubeType, CubeTypeMut, Record, Clone, Copy, Debug, PartialEq)]
#[expand(derive(Clone, Copy))]
struct Row {
    card: CardId,
    weight: f32,
    next: Option<CardId>,
}

/// Five tables in three bindings: one per element type.
#[derive(CubeType, Arena)]
struct Store {
    prices: Table<f32, ReadWrite, (CardId, Day)>,
    totals: AtomicTable<f32, CardId>,
    rows: RecordTable<Row, ReadWrite, CardId>,
    counts: Table<u32, ReadWrite, CardId>,
    weights: Table<f32, ReadOnly, CardId>,
}

#[cube(launch)]
fn kernel(store: &mut Store) {
    let card = CardId(UNIT_POS as u32);
    let row = store.rows.load(card);
    store.prices.store((card, Day(1)), row.weight + store.weights.load(card));
    store.totals.fetch_add(CardId(0), row.weight);
    store.counts.store(card, store.counts.load(card) + 1);
}

fn main() {
    let _ = |client: &Client| {
        let rows = Row::pack(&[Row { card: CardId(0), weight: 1.5, next: None }]);
        let store = StoreArena::new(
            client,
            StoreData {
                prices: ArenaSource::zeros(&[1, 2]),
                totals: ArenaSource::fill(0.0, 1),
                rows: ArenaSource::values(&rows),
                counts: ArenaSource::fill(0, 1),
                weights: ArenaSource::zeros(&[1]),
            },
        );
        store.arg()
    };
}
