//! A blek-shaped workload: an event loop with mutable state updated in branches, and helpers full
//! of transcendental calls inlined at several call sites.
use cubecl::prelude::*;
use std::time::Instant;

#[cfg_attr(feature = "outline", cube)]
#[cfg_attr(not(feature = "outline"), cube(inline))]
fn retrievability(elapsed: f32, stability: f32) -> f32 {
    f32::powf(1.0 + elapsed / (9.0 * stability), -1.0)
}

#[cfg_attr(feature = "outline", cube)]
#[cfg_attr(not(feature = "outline"), cube(inline))]
fn next_stability(stability: f32, difficulty: f32, r: f32, good: bool) -> f32 {
    let mut s = stability;
    if good {
        s *= 1.0
            + f32::exp(1.5)
                * (11.0 - difficulty)
                * f32::powf(s, -0.2)
                * (f32::exp((1.0 - r) * 0.9) - 1.0);
    } else {
        s = 1.9
            * f32::powf(difficulty, -0.1)
            * (f32::powf(s + 1.0, 0.3) - 1.0)
            * f32::exp((1.0 - r) * 2.1);
    }
    s
}

#[cfg_attr(feature = "outline", cube)]
#[cfg_attr(not(feature = "outline"), cube(inline))]
fn next_difficulty(difficulty: f32, r: f32, good: bool) -> f32 {
    let mut d = difficulty;
    if good {
        d -= 0.3 * (1.0 - r);
    } else {
        d += 0.8 * r;
    }
    f32::clamp(d, 1.0, 10.0)
}

#[cfg_attr(feature = "outline", cube)]
#[cfg_attr(not(feature = "outline"), cube(inline))]
fn uniform(state: u32) -> f32 {
    let bits = state * 747796405u32 + 2891336453u32;
    f32::cast_from(bits >> 8u32) * (1.0 / 16777216.0)
}

#[cube(launch)]
fn simulate(
    cards: &[f32],
    out: &mut [f32],
    days: u32,
    #[comptime] siblings: u32,
    #[comptime] salt: u32,
) {
    let id = ABSOLUTE_POS;
    if id < cards.len() {
        let mut stability = cards[id];
        let mut difficulty = 5.0f32;
        let mut due = 1.0f32;
        let mut day = 0u32;
        let mut minutes = 0.0f32;
        let mut open = days > 0;
        let mut rng = id as u32;
        while open {
            let t = f32::cast_from(day);
            if t >= due {
                rng = rng * 1664525u32 + 1013904223u32;
                let u = uniform(rng);
                let r = retrievability(t - due + 1.0, stability);
                let good = u < r;
                let s = next_stability(stability, difficulty, r, good);
                difficulty = next_difficulty(difficulty, r, good);
                stability = s;
                // Siblings reviewed the same day: every helper inlined once more per sibling.
                #[unroll]
                for k in 0..siblings {
                    let v = uniform(rng + k);
                    let rs = retrievability(t - due + 2.0, stability + f32::cast_from(k));
                    let sibling = next_stability(stability + 1.0, difficulty, rs, v < rs);
                    minutes += 0.01 * next_difficulty(sibling, rs, v < rs);
                }
                // Skip past cards that are not due, reading memory in the condition.
                let mut next = id;
                while next < cards.len() && cards[next] <= 0.0 {
                    next += 1;
                }
                minutes += f32::cast_from(next - id);
                let r = retrievability(1.0, stability);
                if r > 0.9 {
                    due = t + stability * 0.5;
                    minutes += 0.2;
                } else {
                    due = t + 1.0;
                    minutes += 0.5 + f32::exp(-r);
                }
            }
            day += 1;
            if day >= days {
                open = false;
            }
        }
        // A constant unique to the run keeps the driver's shader cache from answering.
        out[id] = minutes + f32::cast_from(id as u32 + salt) * 0.0000001;
    }
}

fn main() {
    let device = cubecl::Device::default();
    let client = device.client();
    let n = 1 << 16;
    let cards: Vec<f32> = (0..n).map(|i| 1.0 + (i % 97) as f32).collect();
    let cards = client.create_from_slice(f32::as_bytes(&cards));
    let out = client.empty(n * 4);
    let days: u32 = std::env::args()
        .nth(1)
        .map(|d| d.parse().unwrap())
        .unwrap_or(2000);
    let siblings: u32 = std::env::args()
        .nth(2)
        .map(|d| d.parse().unwrap())
        .unwrap_or(0);
    // A fixed salt (`BENCH_SALT=0`) makes the checksum comparable across runs and builds.
    let salt = std::env::var("BENCH_SALT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        });

    let launch = || {
        simulate::launch(
            &client,
            CubeCount::Static((n as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            unsafe { BufferArg::from_raw_parts(cards.clone(), n) },
            unsafe { BufferArg::from_raw_parts(out.clone(), n) },
            days,
            siblings,
            salt,
        )
    };
    let start = Instant::now();
    launch();
    cubecl::future::block_on(client.sync()).unwrap();
    let cold = start.elapsed();
    let mut warm = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        launch();
        cubecl::future::block_on(client.sync()).unwrap();
        warm.push(start.elapsed());
    }
    warm.sort();
    let bytes = client.read_one(out).unwrap();
    let total: f64 = f32::from_bytes(&bytes).iter().map(|&x| x as f64).sum();
    println!(
        "{} cold {cold:?} warm {:?} checksum {total:.1}",
        client.name(),
        warm[3]
    );
    println!("{}", client.activity());
}
