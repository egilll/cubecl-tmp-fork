use std::sync::{Arc, atomic::AtomicU64};

use crossbeam_utils::CachePadded;

use crate::compute::executable::{Call, Executable};

use crate::compute::threadpool::ThreadTask;

pub struct ComputeTask {
    pub(crate) executable: Executable,
    pub(crate) call: Call,
    pub(crate) next_counter_step: u64,
    pub(crate) atomic_counter: Arc<CachePadded<AtomicU64>>,
}

impl ThreadTask for ComputeTask {
    fn is_ready(&self) -> bool {
        self.atomic_counter
            .load(std::sync::atomic::Ordering::Acquire)
            >= self.next_counter_step
    }
}

impl ComputeTask {
    pub fn compute(&mut self) {
        self.executable.run(&self.call);
        self.atomic_counter
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }
}
