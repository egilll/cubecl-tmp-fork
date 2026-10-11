use crossbeam_utils::CachePadded;
use cubecl_core::CubeDim;
use std::{
    ops::Range,
    sync::{Arc, OnceLock, atomic::AtomicU64},
};

use cubecl_llvm::{PlironData, PlironEngine};

use crate::compute::{
    schedule::BindingsResource,
    threadpool::{
        compute_task::ComputeTask,
        scheduler::{Scheduler, SchedulerVariant},
    },
};

pub mod compute_task;
pub mod scheduler;

trait ThreadTask {
    fn is_ready(&self) -> bool;
}

static INSTANCE: OnceLock<spin::Mutex<Threadpool>> = OnceLock::new();

/// The kernel runner is responsible to manage shared memory as well as threads to execute kernels.
///
/// A single kernel runner is currently used for all kernels.
/// To register work, you have to use the execution queue.
pub struct Threadpool {
    scheduler: Scheduler,
}

impl Threadpool {
    fn init() -> Self {
        let scheduler = Scheduler::new(SchedulerVariant::Dispatcher);

        Self { scheduler }
    }

    /// Resolves the global execution queue instance.
    pub fn get() -> &'static spin::Mutex<Self> {
        INSTANCE.get_or_init(|| spin::Mutex::new(Self::init()))
    }

    /// Queues a launch as calls over ranges of its cubes, and returns how many.
    pub fn execute_data(
        &mut self,
        pliron_engine: PlironEngine,
        bindings: BindingsResource,
        cube_dim: CubeDim,
        cube_count: [u32; 3],
        next_counter_step: u64,
        atomic_counter: &Arc<CachePadded<AtomicU64>>,
    ) -> u64 {
        let whole_cubes = pliron_engine.requirements().whole_cubes;

        let BindingsResource { resources, info } = bindings;
        let buffer_ptrs: Vec<*mut std::ffi::c_void> = resources
            .iter()
            .map(|resource| {
                resource.resource().get_write_ptr_and_length().0 as *mut std::ffi::c_void
            })
            .collect();
        // Pin the resources for the launch's lifetime (see
        // `SharedData::keepalive`).
        let keepalive: Vec<Box<dyn std::any::Any + Send>> = resources
            .into_iter()
            .map(|resource| Box::new(resource) as Box<dyn std::any::Any + Send>)
            .collect();
        let launch = PlironData::new(buffer_ptrs, info.data, cube_count, keepalive);

        let calls = calls(
            cube_count.iter().map(|&count| u64::from(count)).product(),
            cube_dim.num_elems(),
            whole_cubes,
            self.scheduler.workers() * CALLS_PER_WORKER,
        );
        let tasks = calls.len() as u64;
        for (i, (cubes, units)) in calls.into_iter().enumerate() {
            let compute_task = ComputeTask {
                pliron_engine: pliron_engine.clone(),
                pliron_data: launch.call(cubes, units),
                next_counter_step,
                atomic_counter: Arc::clone(atomic_counter),
            };
            self.scheduler.send(i, compute_task);
        }
        tasks
    }
}

/// Calls per worker a launch is split into, so workers that finish early take on more.
const CALLS_PER_WORKER: usize = 4;

/// How a launch of `cubes` cubes of `units` units is split into about `target` calls: ranges of
/// whole cubes, or, with fewer cubes than calls and units that need not run together, ranges of
/// the units of each cube.
fn calls(
    cubes: u64,
    units: u32,
    whole_cubes: bool,
    target: usize,
) -> Vec<(Range<u64>, Range<u32>)> {
    if cubes == 0 || units == 0 {
        return Vec::new();
    }
    let target = target.max(1) as u64;
    if whole_cubes || cubes >= target {
        let calls = cubes.min(target);
        return (0..calls)
            .map(|call| (cubes * call / calls..cubes * (call + 1) / calls, 0..units))
            .collect();
    }
    let per_cube = target.div_ceil(cubes).min(u64::from(units));
    (0..cubes)
        .flat_map(|cube| {
            (0..per_cube).map(move |part| {
                let start = u64::from(units) * part / per_cube;
                let end = u64::from(units) * (part + 1) / per_cube;
                (cube..cube + 1, start as u32..end as u32)
            })
        })
        .collect()
}
