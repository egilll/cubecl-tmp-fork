//! GPU-time accounting for a stream: what each kernel costs, learned from
//! completed command buffers, and what is in flight.
//!
//! Metal times command buffers, not dispatches (`GPUStartTime`/`GPUEndTime`),
//! so a buffer's time is shared among its dispatches by their cube counts.
//! That is coarse for one buffer, but unbiased over the mixes a stream keeps
//! submitting, and enough to bound how long a batch keeps the GPU.

use core::task::Waker;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use cubecl_core::server::TagTime;
use cubecl_environment::sync::Mutex;

/// What a dispatch is estimated by: its kernel and a class of its size.
pub type CostKey = (u64, u8);

/// Estimated cost of a kernel nothing has measured yet.
const UNKNOWN_DISPATCH_MICROS: f64 = 1000.0;
/// Weight of a new measurement in a kernel's running estimate.
const EWMA_WEIGHT: f64 = 0.25;

/// One stream's learned costs and in-flight totals, shared with the
/// completion handlers of its command buffers.
#[derive(Debug, Default)]
pub struct Accounting {
    /// Estimated microseconds per cube, by kernel and size class.
    per_cube_micros: Mutex<HashMap<CostKey, f64>>,
    in_flight_micros: AtomicU64,
    in_flight_batches: AtomicU64,
    waiters: Mutex<Vec<(u64, Waker)>>,
    /// Measured GPU microseconds of completed batches, by label.
    by_label: Mutex<HashMap<&'static str, u64>>,
    /// Measured GPU time of completed batches, by tag.
    by_tag: Mutex<HashMap<&'static str, TagTime>>,
}

/// What a committed command buffer carries, for its completion handler.
#[derive(Debug, Default, Clone)]
pub struct BatchCost {
    /// Each dispatch's key and cube count.
    pub dispatches: Vec<(CostKey, u64)>,
    /// The estimate added to the in-flight total when it was committed.
    pub estimate_micros: u64,
    /// The label its launches were made under.
    pub label: Option<&'static str>,
    /// The tag its launches carried.
    pub tag: Option<&'static str>,
}

/// The size class of a dispatch of `cubes` cubes: its power of two, so a
/// kernel's per-cube cost is learned separately for very different sizes.
pub fn size_class(cubes: u64) -> u8 {
    (u64::BITS - cubes.max(1).leading_zeros()) as u8
}

impl Accounting {
    /// Estimated GPU time of a dispatch, in microseconds.
    pub fn estimate(&self, key: CostKey, cubes: u64) -> f64 {
        match self.per_cube_micros.lock().get(&key) {
            Some(per_cube) => per_cube * cubes as f64,
            None => UNKNOWN_DISPATCH_MICROS,
        }
    }

    /// A batch with `estimate_micros` of work was committed.
    pub fn committed(&self, estimate_micros: u64) {
        self.in_flight_micros
            .fetch_add(estimate_micros, Ordering::AcqRel);
        self.in_flight_batches.fetch_add(1, Ordering::AcqRel);
    }

    /// A committed batch completed, having taken `gpu_micros` on the GPU
    /// (`None` when Metal gave no timestamps).
    pub fn completed(&self, batch: &BatchCost, gpu_micros: Option<f64>) {
        if let (Some(gpu_micros), Some(label)) = (gpu_micros, batch.label) {
            *self.by_label.lock().entry(label).or_default() += gpu_micros.round() as u64;
        }
        if let (Some(gpu_micros), Some(tag)) = (gpu_micros, batch.tag) {
            let cubes: u64 = batch.dispatches.iter().map(|(_, cubes)| cubes).sum();
            let mut tags = self.by_tag.lock();
            let time = tags.entry(tag).or_default();
            time.micros += gpu_micros;
            time.cubes += cubes;
            time.batches += 1;
            time.max_micros = time.max_micros.max(gpu_micros);
        }
        if let Some(gpu_micros) = gpu_micros {
            let cubes: u64 = batch.dispatches.iter().map(|(_, cubes)| cubes).sum();
            if cubes > 0 && gpu_micros > 0.0 {
                let per_cube = gpu_micros / cubes as f64;
                let mut costs = self.per_cube_micros.lock();
                for (key, _) in &batch.dispatches {
                    costs
                        .entry(*key)
                        .and_modify(|old| *old += EWMA_WEIGHT * (per_cube - *old))
                        .or_insert(per_cube);
                }
            }
        }
        let remaining = self
            .in_flight_micros
            .fetch_sub(batch.estimate_micros, Ordering::AcqRel)
            .saturating_sub(batch.estimate_micros);
        self.in_flight_batches.fetch_sub(1, Ordering::AcqRel);
        // Wake whoever waited for less than what remains now.
        let mut waiters = self.waiters.lock();
        waiters.retain(|(threshold, waker)| {
            let below = remaining < *threshold;
            if below {
                waker.wake_by_ref();
            }
            !below
        });
    }

    /// Measured GPU microseconds by label so far.
    pub fn by_label(&self) -> Vec<(&'static str, u64)> {
        self.by_label
            .lock()
            .iter()
            .map(|(label, micros)| (*label, *micros))
            .collect()
    }

    /// Measured GPU time so far, by tag.
    pub fn by_tag(&self) -> Vec<(&'static str, TagTime)> {
        self.by_tag
            .lock()
            .iter()
            .map(|(tag, time)| (*tag, *time))
            .collect()
    }

    /// Microseconds and batches in flight.
    pub fn in_flight(&self) -> (u64, u64) {
        (
            self.in_flight_micros.load(Ordering::Acquire),
            self.in_flight_batches.load(Ordering::Acquire),
        )
    }

    /// Wake `waker` once less than `threshold` microseconds are in flight.
    pub fn wake_below(&self, threshold: u64, waker: Waker) {
        self.waiters.lock().push((threshold, waker));
    }
}

/// A future resolving once less than `threshold` microseconds of estimated
/// GPU time are in flight on a stream.
pub struct Below {
    pub accounting: std::sync::Arc<Accounting>,
    pub threshold: u64,
}

impl core::future::Future for Below {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.accounting.in_flight().0 < self.threshold {
            return core::task::Poll::Ready(());
        }
        self.accounting
            .wake_below(self.threshold, cx.waker().clone());
        // A completion between the check and the registration would have
        // found no waiter: check again.
        match self.accounting.in_flight().0 < self.threshold {
            true => core::task::Poll::Ready(()),
            false => core::task::Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn an_unmeasured_kernel_costs_the_default_then_learns() {
        let accounting = Accounting::default();
        let key = (7, size_class(64));
        assert_eq!(accounting.estimate(key, 64), UNKNOWN_DISPATCH_MICROS);

        let batch = BatchCost {
            dispatches: vec![(key, 64), (key, 64)],
            estimate_micros: 2000,
            label: None,
            tag: None,
        };
        accounting.committed(batch.estimate_micros);
        assert_eq!(accounting.in_flight(), (2000, 1));
        // 128 cubes took 256 µs: 2 µs per cube.
        accounting.completed(&batch, Some(256.0));
        assert_eq!(accounting.in_flight(), (0, 0));
        assert_eq!(accounting.estimate(key, 64), 128.0);

        // Later measurements move the estimate, not replace it.
        accounting.committed(128);
        accounting.completed(
            &BatchCost {
                dispatches: vec![(key, 64)],
                estimate_micros: 128,
                label: None,
                tag: None,
            },
            Some(64.0 * 6.0),
        );
        assert_eq!(accounting.estimate(key, 1), 2.0 + EWMA_WEIGHT * (6.0 - 2.0));
    }

    #[test]
    fn waiting_for_less_in_flight_wakes_on_completion() {
        let accounting = Arc::new(Accounting::default());
        let batch = BatchCost {
            dispatches: vec![],
            estimate_micros: 500,
            label: None,
            tag: None,
        };
        accounting.committed(500);
        let mut below = core::pin::pin!(Below {
            accounting: accounting.clone(),
            threshold: 100,
        });
        let woken = Arc::new(std::sync::atomic::AtomicBool::new(false));
        struct Flag(Arc<std::sync::atomic::AtomicBool>);
        impl std::task::Wake for Flag {
            fn wake(self: Arc<Self>) {
                self.0.store(true, Ordering::Release);
            }
        }
        let waker = std::task::Waker::from(Arc::new(Flag(woken.clone())));
        let mut cx = core::task::Context::from_waker(&waker);
        assert!(below.as_mut().poll(&mut cx).is_pending());
        accounting.completed(&batch, None);
        assert!(woken.load(Ordering::Acquire));
        assert!(below.as_mut().poll(&mut cx).is_ready());
    }

    #[test]
    fn size_classes_are_powers_of_two() {
        assert_eq!(size_class(0), size_class(1));
        assert_eq!(size_class(2), size_class(3));
        assert_ne!(size_class(3), size_class(4));
    }
}
