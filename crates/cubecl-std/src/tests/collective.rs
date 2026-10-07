use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_runtime::runtime::Runtime;

use crate::collective::*;

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr, Debug)]
#[device_repr(copy)]
struct Score(f32);

#[repr(transparent)]
#[derive(CubeType, CubeTypeMut, DeviceRepr, Debug)]
#[device_repr(copy, key)]
struct Item(u32);

#[cube(launch)]
fn kernel_collectives(ranks: &[f32], output: &mut [f32]) {
    let unit = UNIT_POS as usize;
    let mut sums = SharedStorage::<Score>::new(comptime![subgroups(128)]);
    let mut best_ranks = SharedStorage::<Score>::new(comptime![subgroups(128)]);
    let mut best_items = SharedStorage::<Item>::new(comptime![subgroups(128)]);
    let total = cube_sum(Score(ranks[unit]), &mut sums);
    let best = cube_argmax(Best::<Score, Item> { rank: Score(ranks[unit]), index: Item(UNIT_POS) }, &mut best_ranks, &mut best_items);
    let bits = plane_or(1u32 << (UNIT_POS % 8));
    output[4 * unit] = total.0;
    output[4 * unit + 1] = best.rank.0;
    output[4 * unit + 2] = f32::cast_from(best.index.0);
    output[4 * unit + 3] = f32::cast_from(bits);
}

/// Every unit of a 128-unit cube receives the sum and the best offer
/// (ties to the lower index), and its subgroup's OR.
pub fn test_collectives<R: Runtime>(client: Client) {
    let ranks: Vec<f32> = (0..128).map(|i| if i == 90 || i == 17 { 500.0 } else { ((i * 37) % 101) as f32 }).collect();
    let input = Buffer::create(&client, &ranks);
    let output = Buffer::<f32>::empty(&client, 4 * 128);
    kernel_collectives::launch(&client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(128), (&input).into(), (&output).into());
    let output = output.read(&client).unwrap();
    let total: f32 = ranks.iter().sum();
    for unit in output.chunks(4) {
        assert!((unit[0] - total).abs() < 1e-2 * total.abs());
        assert_eq!(unit[1], 500.0);
        assert_eq!(unit[2], 17.0, "ties go to the lower index");
        assert_eq!(unit[3], 255.0);
    }
}

#[macro_export]
macro_rules! testgen_collective {
    () => {
        #[$crate::tests::test_log::test]
        fn test_collectives() {
            let client = TestRuntime::client(&Default::default());
            cubecl_std::tests::collective::test_collectives::<TestRuntime>(client);
        }
    };
}
