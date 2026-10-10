use crate::compute::copies::MetalCopies;
use crate::memory::MetalStorage;
use cubecl_core::server::ExecutionFaultKind;
use cubecl_core::{MemoryConfiguration, server::ServerError};
use cubecl_environment::sync::Mutex;
use cubecl_ir::MemoryDeviceProperties;
use cubecl_server::memory_management::relocation::RelocationReason;
use cubecl_server::{
    logging::ServerLogger,
    memory_management::{ErrorGraph, FailureId, MemoryManagement, MemoryManagementOptions},
    server::BufferBinding,
    stream::{EventStreamBackend, StreamMemory},
    timestamp_profiler::TimestampProfiler,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferError, MTLCommandBufferStatus, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLSharedEvent,
};
use std::ptr::NonNull;

use crate::compute::accounting::{Accounting, BatchCost};
use std::sync::Arc;

/// Active encoder state for batching multiple kernel dispatches.
pub struct ActiveEncoder {
    pub command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    /// Temporary buffers that must stay alive until this encoder's work completes.
    pub temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// What the dispatches since the last barrier touch.
    pub hazards: Hazards,
    /// This command buffer's place in the stream's order, see
    /// [`FaultState::confirmed`].
    pub seq: u64,
    /// Report a fault when it completes, for tests of recovery: a real GPU
    /// fault can't be caused on demand. Only tests set it.
    pub inject_fault: bool,
    /// The dispatches it carries, for GPU-time accounting.
    pub cost: BatchCost,
}

/// A byte range of one `MTLBuffer`, keyed by the buffer's address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Access {
    pub buffer: usize,
    pub range: core::ops::Range<u64>,
}

impl Access {
    fn overlaps(&self, other: &Access) -> bool {
        self.buffer == other.buffer
            && self.range.start < other.range.end
            && other.range.start < self.range.end
    }
}

/// The ranges the dispatches encoded since the last barrier read and write.
///
/// The encoder is concurrent: Metal lets its dispatches overlap and tracks no
/// hazard between them, so a dispatch that reads what an earlier one writes,
/// or writes what an earlier one reads or writes, needs a barrier first.
/// Ranges are physical (buffer and bytes), not tensors: two tensors carved
/// from one page don't conflict, and memory a freed tensor hands to a new one
/// still does.
#[derive(Debug, Default)]
pub struct Hazards {
    reads: Vec<Access>,
    writes: Vec<Access>,
}

impl Hazards {
    /// Records a dispatch's accesses. Returns whether it must wait for the
    /// dispatches before it, in which case those are forgotten: the barrier
    /// orders everything after it behind them.
    pub fn record(&mut self, reads: &[Access], writes: &[Access]) -> bool {
        let conflicts = reads
            .iter()
            .any(|read| self.writes.iter().any(|w| w.overlaps(read)))
            || writes.iter().any(|write| {
                self.writes
                    .iter()
                    .chain(&self.reads)
                    .any(|other| other.overlaps(write))
            });
        if conflicts {
            self.reads.clear();
            self.writes.clear();
        }
        self.reads.extend_from_slice(reads);
        self.writes.extend_from_slice(writes);
        conflicts
    }
}

/// What a stream's completion handlers report back.
#[derive(Debug, Default)]
pub struct FaultState {
    /// The first failure, sticky until the stream is reset.
    pub slot: Mutex<Option<StreamFault>>,
    /// The sequence number of the last command buffer that completed with no
    /// fault before it: the writes of every batch up to it landed.
    pub confirmed: core::sync::atomic::AtomicU64,
}

/// The first failure a stream's command buffers reported.
#[derive(Debug, Clone)]
pub struct StreamFault {
    pub kind: ExecutionFaultKind,
    pub reason: String,
}

impl StreamFault {
    /// Classify a failed command buffer from its `MTLCommandBufferError` code
    /// and its messages. The causes macOS reports through IOGPU
    /// (`kIOGPUCommandBufferCallbackError…`) only appear in the error's
    /// description and underlying error, so those are matched by name.
    pub(crate) fn classify(code: isize, description: &str, debug: &str) -> Self {
        let text = format!("{description} {debug}");
        let has = |needles: &[&str]| needles.iter().any(|needle| text.contains(needle));
        let kind = if has(&["ImpactingInteractivity", "Impacting Interactivity"]) {
            ExecutionFaultKind::Interactivity
        } else if has(&["InnocentVictim", "Innocent Victim"]) {
            ExecutionFaultKind::InnocentVictim
        } else if code == MTLCommandBufferError::PageFault.0 as isize || has(&["PageFault"]) {
            ExecutionFaultKind::PageFault
        } else if code == MTLCommandBufferError::Timeout.0 as isize || has(&["Hang", "Timeout"]) {
            ExecutionFaultKind::Timeout
        } else if code == MTLCommandBufferError::OutOfMemory.0 as isize {
            ExecutionFaultKind::OutOfMemory
        } else {
            ExecutionFaultKind::Unknown
        };
        Self {
            kind,
            reason: description.to_string(),
        }
    }

    pub fn into_error(self) -> ServerError {
        ServerError::ExecutionFault {
            kind: self.kind,
            reason: self.reason,
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        }
    }
}

/// Installs a completion handler that drops `temporaries` and, on a failed
/// command buffer, records the fault on the stream's sticky slot. `signal_event`
/// is `Some` when the buffer signals an event; it is forced on failure so
/// dependent waiters return promptly — and fail, because every wait checks the
/// fault slot after its event lands.
fn install_completion_handler(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    signal_event: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    state: Arc<FaultState>,
    seq: u64,
    injected: bool,
) {
    install_completion_handler_signaling(
        command_buffer,
        temporaries,
        signal_event,
        state,
        seq,
        injected,
        false,
    )
}

/// [`install_completion_handler`], which with `signal_on_success` also
/// signals the event from the handler when the buffer succeeds, instead of
/// from the GPU. Completion handlers of one queue run in order, so a wait on
/// such an event also sees every fault an earlier buffer of the queue hit.
fn install_completion_handler_signaling(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    signal_event: Option<(Retained<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
    state: Arc<FaultState>,
    seq: u64,
    injected: bool,
    signal_on_success: bool,
) {
    let temporaries = Mutex::new(Some(temporaries));
    let block = block2::RcBlock::new(
        move |cmd_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let _ = temporaries.lock().take();

            let cmd_buf = unsafe { cmd_buf.as_ref() };
            if cmd_buf.status() != MTLCommandBufferStatus::Error && !injected {
                // Buffers complete in queue order, so everything up to this
                // one ran, unless an earlier one already faulted.
                if state.slot.lock().is_none() {
                    state
                        .confirmed
                        .fetch_max(seq, core::sync::atomic::Ordering::AcqRel);
                }
                if signal_on_success && let Some((event, value)) = &signal_event {
                    event.setSignaledValue(*value);
                }
            } else {
                let fault = match cmd_buf.error().filter(|_| !injected) {
                    Some(err) => StreamFault::classify(
                        err.code(),
                        &format!("{}", err.localizedDescription()),
                        &format!("{err:?}"),
                    ),
                    None => StreamFault {
                        kind: ExecutionFaultKind::Unknown,
                        reason: match injected {
                            true => "a fault injected by a test".to_string(),
                            false => {
                                "Metal command buffer failed with an unknown error".to_string()
                            }
                        },
                    },
                };
                log::warn!(
                    "Metal command buffer failed ({}): {}",
                    fault.kind,
                    fault.reason
                );

                // A fault at execution time can name no buffer: the work's
                // claims were released at enqueue — a claim covers enqueue,
                // not execution — and this handler holds only the event and
                // the staging temporaries. So the fault is recorded on the
                // stream instead, sticky, and every later wait on it fails:
                // reads report the fault instead of returning garbage, and
                // writes taint their destinations with it. First fault wins,
                // and none is ever cleared — clearing is exactly how stale
                // bytes would start reading clean again.
                let mut slot = state.slot.lock();
                if slot.is_none() {
                    *slot = Some(fault);
                }

                // Metal leaves encoded events unsignaled on fault; signal
                // manually so a dependent wait returns promptly — with the
                // fault recorded above.
                if let Some((event, value)) = &signal_event {
                    event.setSignaledValue(*value);
                }
            }
        },
    );

    // SAFETY: `addCompletedHandler` copies the block, so the pointer need not outlive
    // this call. The raw-pointer form bypasses block2's `Send` bound, but everything the
    // block touches on the Metal completion thread is thread-safe: `Retained` drops via
    // atomic Obj-C `release`, `setSignaledValue` is an atomic write, the temporaries are
    // behind a `Mutex`, and `log` is `Sync`.
    unsafe {
        command_buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block) as *mut _);
    }
}

/// Metal stream with its own command queue and memory management.
pub struct MetalStream {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub memory_management: MemoryManagement<MetalStorage>,
    /// Encoder for the current dispatch batch, `None` between batches.
    pub active_encoder: Option<ActiveEncoder>,
    pub batch_ops: usize,
    pub batch_bytes: usize,
    pub shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    /// Next event signal value.
    pub event_counter: u64,
    /// Device-specific batch flush thresholds.
    pub max_ops_per_batch: usize,
    pub max_mb_per_batch: usize,
    /// Ops submitted without a GPU wait, used for back-pressure.
    pub submitted_ops: usize,
    /// Max submitted ops before we wait on the GPU to drain.
    pub max_submitted_ops: usize,
    /// Last committed command buffer, kept alive for back-pressure waits.
    pub last_command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// When `Some`, device profiling is active on this stream: each work-bearing command
    /// buffer committed during the window is collected here so its GPU timestamps
    /// (`GPUStartTime`/`GPUEndTime`) can be read after completion.
    pub profiling: Option<Vec<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    /// The profiling windows open on this stream, which a failure on it invalidates.
    pub timestamps: TimestampProfiler,
    /// The first GPU-time fault a completed command buffer reported, sticky
    /// for the stream's life. Shared with every completion handler and every
    /// [`MetalEvent`], whose waits fail on it — see
    /// [`install_completion_handler`] for why the fault lives here and not on
    /// a buffer.
    pub fault: Arc<FaultState>,
    /// The sequence number of the last command buffer opened.
    pub batch_seq: u64,
    /// What launches wrote, by the sequence number of the command buffer that
    /// carried them, until a completion confirms it: what a fault leaves
    /// unwritten, so what [`MetalStream::reset`] fails.
    pub unconfirmed_writes: Vec<(u64, BufferBinding)>,
    /// The event value each recently flushed batch signals, by sequence
    /// number, so a host access can wait for just the batch it needs.
    pub batch_events: std::collections::VecDeque<(u64, u64)>,
    /// Learned kernel costs and what is in flight, shared with the
    /// completion handlers.
    pub accounting: Arc<Accounting>,
    /// Estimated GPU microseconds of the open batch.
    pub batch_estimate: f64,
    /// Commit the open batch once its estimate reaches this; see
    /// `StreamingConfig::max_batch_gpu_micros`.
    pub max_batch_gpu_micros: Option<u64>,
    /// The label launches are attributed to; a batch carries one label.
    pub label: Option<&'static str>,
    /// The tag launches carry; a batch carries one tag.
    pub tag: Option<&'static str>,
}

impl std::fmt::Debug for MetalStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalStream")
            .field("has_active_encoder", &self.active_encoder.is_some())
            .field("batch_ops", &self.batch_ops)
            .field("batch_bytes", &self.batch_bytes)
            .field("event_counter", &self.event_counter)
            .finish()
    }
}

impl StreamMemory for MetalStream {
    fn failure(&self, binding: &BufferBinding) -> Option<FailureId> {
        self.memory_management
            .failure(&binding.memory, binding.range())
    }

    fn taint(&mut self, binding: &BufferBinding, failure: FailureId, failures: &mut ErrorGraph) {
        self.memory_management
            .taint(&binding.memory, binding.range(), failure, failures)
    }

    fn written(&mut self, binding: &BufferBinding, failures: &mut ErrorGraph) {
        self.memory_management
            .written(&binding.memory, binding.range(), failures)
    }
}

impl MetalStream {
    /// Returns the active batch encoder, creating one if none is open.
    pub fn get_or_create_encoder(&mut self) -> &mut ActiveEncoder {
        if self.active_encoder.is_none() {
            let command_buffer = (*self.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            // Concurrent: dispatches overlap unless a barrier orders them,
            // which the launch inserts where [`Hazards`] finds a conflict.
            let encoder = (*command_buffer)
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
                .expect("Failed to create compute command encoder");

            self.batch_seq += 1;
            self.active_encoder = Some(ActiveEncoder {
                command_buffer,
                encoder,
                temporaries: Vec::new(),
                hazards: Hazards::default(),
                seq: self.batch_seq,
                inject_fault: false,
                cost: BatchCost::default(),
            });
        }

        self.active_encoder.as_mut().unwrap()
    }

    /// Commit the batch this stream has open and wait until everything it
    /// submitted has run.
    pub fn finish(&mut self) {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

        if let Some(active) = self.active_encoder.take() {
            (*active.encoder).endEncoding();
            self.account_batch(&active);
            install_completion_handler(
                &active.command_buffer,
                active.temporaries,
                None,
                self.fault.clone(),
                active.seq,
                active.inject_fault,
            );
            (*active.command_buffer).commit();
            // As a flush does: an open profile measures the dispatches this
            // buffer carries.
            if self.batch_ops > 0
                && let Some(buffers) = self.profiling.as_mut()
            {
                buffers.push(active.command_buffer.clone());
            }
            self.last_command_buffer = Some(active.command_buffer);
        }
        self.batch_ops = 0;
        self.batch_bytes = 0;
        if let Some(command_buffer) = self.last_command_buffer.take() {
            (*command_buffer).waitUntilCompleted();
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        }
        // Everything submitted has run, so nothing is left to regulate.
        self.submitted_ops = 0;
    }

    /// Record that the open command buffer writes `bindings`.
    pub fn note_writes(&mut self, bindings: impl IntoIterator<Item = BufferBinding>) {
        let seq = self.batch_seq;
        self.forget_confirmed_writes();
        self.unconfirmed_writes
            .extend(bindings.into_iter().map(|binding| (seq, binding)));
    }

    /// Drop the writes of batches known to have completed. Each entry holds
    /// a binding, which keeps its allocation bound: a buffer freed by every
    /// handle must not stay reserved until later writes push it out.
    pub fn forget_confirmed_writes(&mut self) {
        let confirmed = self
            .fault
            .confirmed
            .load(core::sync::atomic::Ordering::Acquire);
        self.unconfirmed_writes.retain(|(s, _)| *s > confirmed);
    }

    /// Count the batch `active` is about to commit as in flight, and learn
    /// from it once it completes.
    pub fn account_batch(&mut self, active: &ActiveEncoder) {
        let mut cost = active.cost.clone();
        cost.estimate_micros = self.batch_estimate.round() as u64;
        cost.label = self.label;
        cost.tag = self.tag;
        self.batch_estimate = 0.0;
        self.accounting.committed(cost.estimate_micros);
        let accounting = self.accounting.clone();
        let block = block2::RcBlock::new(
            move |cmd_buf: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let cmd_buf = unsafe { cmd_buf.as_ref() };
                let gpu = (cmd_buf.GPUEndTime() - cmd_buf.GPUStartTime()) * 1e6;
                let measured = (cmd_buf.status() == MTLCommandBufferStatus::Completed && gpu > 0.0)
                    .then_some(gpu);
                accounting.completed(&cost, measured);
            },
        );
        // SAFETY: as in `install_completion_handler`: the block only touches
        // the accounting's atomics and mutexes.
        unsafe {
            active
                .command_buffer
                .addCompletedHandler(block2::RcBlock::as_ptr(&block) as *mut _);
        }
    }

    /// Copy `data` into `target` in the open batch: after every dispatch
    /// encoded before it, before every dispatch encoded after it, and without
    /// waiting for the device. Each piece copies `data[source]` to the byte
    /// `destination` of `target`.
    ///
    /// The bytes travel through a staging buffer the batch keeps alive, so
    /// the host never touches memory a pending dispatch may still read. The
    /// blit gets an encoder of its own; tracked resources order it against
    /// the compute encoders on either side.
    pub fn upload(
        &mut self,
        target: &ProtocolObject<dyn MTLBuffer>,
        data: &[u8],
        pieces: impl IntoIterator<Item = (core::ops::Range<usize>, usize)>,
    ) -> Result<(), ServerError> {
        use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder, MTLResourceOptions};

        let staging = NonNull::new(data.as_ptr() as *mut core::ffi::c_void)
            .and_then(|bytes| unsafe {
                (*self.device).newBufferWithBytes_length_options(
                    bytes,
                    data.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .ok_or_else(|| ServerError::Generic {
                reason: format!(
                    "failed to allocate a {} B staging buffer for a write",
                    data.len()
                ),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            })?;
        let active = self.get_or_create_encoder();
        (*active.encoder).endEncoding();
        let blit = (*active.command_buffer)
            .blitCommandEncoder()
            .expect("Failed to create blit command encoder");
        for (source, destination) in pieces {
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &staging,
                    source.start,
                    target,
                    destination,
                    source.len(),
                );
            }
        }
        blit.endEncoding();
        active.encoder = (*active.command_buffer)
            .computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            .expect("Failed to create compute command encoder");
        // The new encoder starts after the blit; nothing before it can race.
        active.hazards = Hazards::default();
        active.temporaries.push(staging);
        self.batch_ops += 1;
        self.batch_bytes += data.len();
        Ok(())
    }

    /// Whether the open batch's estimated GPU time reached the configured
    /// bound, so it should be committed now.
    pub fn batch_over_budget(&self) -> bool {
        self.max_batch_gpu_micros
            .is_some_and(|max| self.batch_estimate >= max as f64)
    }

    /// Copy each `(buffer, offset, size)` region into a fresh shared staging
    /// buffer, in a command buffer committed after everything this stream
    /// submitted so far, and return the event that signals the copies.
    pub fn copy_to_staging(
        &mut self,
        regions: &[(crate::memory::MetalBufferHandle, u64, u64)],
    ) -> Result<(MetalEvent, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>), ServerError> {
        use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder, MTLResourceOptions};

        let command_buffer = (*self.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");
        let blit = command_buffer
            .blitCommandEncoder()
            .expect("Failed to create blit command encoder");
        let mut stagings = Vec::with_capacity(regions.len());
        for (resource, offset, size) in regions {
            let staging = (*self.device)
                .newBufferWithLength_options(*size as usize, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| ServerError::Generic {
                    reason: format!("failed to allocate a {size} B staging buffer for a read"),
                    backtrace: cubecl_environment::backtrace::BackTrace::capture(),
                })?;
            let source: &ProtocolObject<dyn MTLBuffer> = resource.inner().as_ref();
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    source,
                    *offset as usize,
                    &staging,
                    0,
                    *size as usize,
                );
            }
            stagings.push(staging);
        }
        blit.endEncoding();

        // Signaled by the completion handler rather than the GPU: handlers
        // run in queue order, so by then a fault of any earlier buffer of
        // this stream is recorded, and the read fails on it instead of
        // returning bytes the faulted work never wrote.
        self.event_counter += 1;
        let value = self.event_counter;
        install_completion_handler_signaling(
            &command_buffer,
            Vec::new(),
            Some((self.shared_event.clone(), value)),
            self.fault.clone(),
            0,
            false,
            true,
        );
        command_buffer.commit();
        self.last_command_buffer = Some(command_buffer);
        Ok((
            MetalEvent::new(self.shared_event.clone(), value, self.fault.clone()),
            stagings,
        ))
    }

    /// The newest command buffer that writes into any of `bindings` and
    /// hasn't completed yet, if any.
    pub fn pending_writer(&self, bindings: &[BufferBinding]) -> Option<u64> {
        let confirmed = self
            .fault
            .confirmed
            .load(core::sync::atomic::Ordering::Acquire);
        self.unconfirmed_writes
            .iter()
            .filter(|(seq, _)| *seq > confirmed)
            .filter(|(_, written)| bindings.iter().any(|binding| overlaps(written, binding)))
            .map(|(seq, _)| *seq)
            .max()
    }

    /// The event value the flushed batch `seq` signals, if it is still known.
    pub fn event_for_batch(&self, seq: u64) -> Option<u64> {
        self.batch_events
            .iter()
            .find(|(batch, _)| *batch == seq)
            .map(|(_, value)| *value)
    }

    /// Recover from an execution fault: wait for what was submitted, then
    /// start over with a new queue and an empty fault slot. Returns the fault
    /// and the buffers whose writes it may have lost, which the caller fails,
    /// or `None` when the stream has no fault.
    pub fn reset(&mut self) -> Option<(StreamFault, Vec<BufferBinding>)> {
        let fault = self.fault.slot.lock().clone()?;
        self.finish();
        let confirmed = self
            .fault
            .confirmed
            .load(core::sync::atomic::Ordering::Acquire);
        let lost = core::mem::take(&mut self.unconfirmed_writes)
            .into_iter()
            .filter(|(seq, _)| *seq > confirmed)
            .map(|(_, binding)| binding)
            .collect();
        self.queue = (*self.device)
            .newCommandQueue()
            .expect("Failed to create command queue");
        // Handlers still in flight hold the old state; nothing new reads it.
        self.fault = Arc::new(FaultState::default());
        Some((fault, lost))
    }

    /// Empty the outdated pools into the current pages.
    ///
    /// The caller has [finished](Self::finish) every stream first: the blits
    /// that move the bytes follow every dispatch that could still read them.
    pub fn relocate(&mut self, reason: RelocationReason, failures: &mut ErrorGraph) {
        let mut copier = MetalCopies::new(self.queue.clone());
        self.memory_management
            .relocate(&mut copier, reason, failures);
    }

    /// Waits on a previously submitted command buffer if total queued ops
    /// exceed `max_submitted_ops`, then resets the counter. The memory is
    /// cleaned up by its reservations, not here.
    pub fn regulate(&mut self, ops_in_batch: usize) {
        self.submitted_ops += ops_in_batch;

        if self.submitted_ops >= self.max_submitted_ops {
            if let Some(cmd_buf) = self.last_command_buffer.take() {
                (*cmd_buf).waitUntilCompleted();
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            }
            self.submitted_ops = 0;
        }
    }
}

/// Whether two bindings share bytes of one allocation.
fn overlaps(a: &BufferBinding, b: &BufferBinding) -> bool {
    let (id_a, start_a, end_a) = a.claim_key();
    let (id_b, start_b, end_b) = b.claim_key();
    id_a == id_b && start_a < end_b && start_b < end_a
}

/// Metal event for synchronization using `MTLSharedEvent`.
#[derive(Clone)]
pub struct MetalEvent {
    shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub value: u64,
    /// The stream's sticky fault slot, checked after every wait: a forced
    /// event completes the wait, and this is what fails it.
    fault: Arc<FaultState>,
}

// SAFETY: MTLSharedEvent's signaledValue is atomically updated by the GPU.
unsafe impl Send for MetalEvent {}

impl MetalEvent {
    pub fn new(
        shared_event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
        value: u64,
        fault: Arc<FaultState>,
    ) -> Self {
        Self {
            shared_event,
            value,
            fault,
        }
    }

    /// Check if the event has been signaled (non-blocking).
    pub fn is_complete(&self) -> bool {
        (*self.shared_event).signaledValue() >= self.value
    }

    /// Block until the event is signaled.
    ///
    /// # Errors
    ///
    /// A timeout, and the stream's recorded GPU-time fault: a faulted command
    /// buffer force-signals its event so the wait itself returns, and this
    /// check is what turns that into the failure every dependent caller has
    /// to hear — a read reports it instead of returning garbage, a write
    /// taints its destinations with it.
    pub fn wait_sync(self) -> Result<(), ServerError> {
        let timeout_ms = 60_000;
        let result = (*self.shared_event).waitUntilSignaledValue_timeoutMS(self.value, timeout_ms);
        if !result {
            return Err(ServerError::Generic {
                reason: "Metal event wait timed out".to_string(),
                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
            });
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        if let Some(fault) = self.fault.slot.lock().clone() {
            return Err(fault.into_error());
        }
        Ok(())
    }

    /// Resolves once the event is signaled, without blocking a thread: the
    /// event's listener wakes the future. Fails as [`wait_sync`](Self::wait_sync)
    /// does when the stream faulted.
    pub fn completion(self) -> impl core::future::Future<Output = Result<(), ServerError>> + Send {
        EventCompletion {
            event: self,
            state: Arc::new(Mutex::new(ListenState::default())),
        }
    }

    pub fn wait_async(self, stream: &mut MetalStream) {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLEvent};

        if std::ptr::eq(
            &*self.shared_event as *const _,
            &*stream.shared_event as *const _,
        ) {
            return;
        }

        if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();
            stream.account_batch(&active);
            install_completion_handler(
                &active.command_buffer,
                active.temporaries,
                None,
                stream.fault.clone(),
                active.seq,
                active.inject_fault,
            );
            (*active.command_buffer).commit();
        }

        let command_buffer = (*stream.queue)
            .commandBuffer()
            .expect("Failed to create command buffer");

        let event_ref: &ProtocolObject<dyn MTLEvent> =
            ProtocolObject::from_ref(&*self.shared_event);
        (*command_buffer).encodeWaitForEvent_value(event_ref, self.value);
        (*command_buffer).commit();
    }
}

#[derive(Default)]
struct ListenState {
    registered: bool,
    waker: Option<core::task::Waker>,
}

struct EventCompletion {
    event: MetalEvent,
    state: Arc<Mutex<ListenState>>,
}

// SAFETY: as `MetalEvent`; the listener state is behind a mutex.
unsafe impl Send for EventCompletion {}

impl core::future::Future for EventCompletion {
    type Output = Result<(), ServerError>;

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        let this = self.get_mut();
        if this.event.is_complete() {
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            return core::task::Poll::Ready(match this.event.fault.slot.lock().clone() {
                Some(fault) => Err(fault.into_error()),
                None => Ok(()),
            });
        }
        let mut state = this.state.lock();
        state.waker = Some(cx.waker().clone());
        if !state.registered {
            state.registered = true;
            let shared = this.state.clone();
            let block = block2::RcBlock::new(
                move |_: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _: u64| {
                    if let Some(waker) = shared.lock().waker.take() {
                        waker.wake();
                    }
                },
            );
            let listener = objc2_metal::MTLSharedEventListener::new();
            // SAFETY: Metal copies the block; it only touches the mutex-held
            // waker, from Metal's notification queue. A value already reached
            // notifies at once.
            unsafe {
                this.event.shared_event.notifyListener_atValue_block(
                    &listener,
                    this.event.value,
                    block2::RcBlock::as_ptr(&block) as *mut _,
                );
            }
        }
        drop(state);
        // The event may have been signaled between the check and the
        // registration; the listener then fires at once, but check again so
        // a missed wake can't strand the future.
        if this.event.is_complete() {
            cx.waker().wake_by_ref();
        }
        core::task::Poll::Pending
    }
}

/// Backend for creating Metal streams
#[derive(Debug)]
pub struct MetalStreamBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    logger: Arc<ServerLogger>,
}

impl MetalStreamBackend {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
    ) -> Self {
        Self {
            device,
            mem_props,
            mem_config,
            logger,
        }
    }
}

impl EventStreamBackend for MetalStreamBackend {
    type Stream = MetalStream;
    type Event = MetalEvent;

    fn create_stream(&self) -> Result<Self::Stream, ServerError> {
        let queue = (*self.device)
            .newCommandQueue()
            .expect("Failed to create command queue");

        let shared_event = (*self.device)
            .newSharedEvent()
            .expect("Failed to create shared event");

        let storage = MetalStorage::new(self.device.clone());

        let memory_management = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
            self.logger.clone(),
            MemoryManagementOptions::new("Metal GPU Memory"),
        );

        // Tier batch limits by GPU architecture: the architecture name's last
        // character encodes the tier ('p' phone, 'g' base/pro, 's' max, 'd' ultra).
        let arch = (*self.device).architecture().name().to_string();
        let (max_ops_per_batch, max_mb_per_batch, max_submitted_ops) = match arch.chars().last() {
            Some('s' | 'd') => (50, 50, 512), // max, ultra
            Some('p') => (20, 20, 256),       // phone
            _ => (40, 40, 512),               // base, pro, and unrecognized
        };

        Ok(MetalStream {
            device: self.device.clone(),
            queue,
            memory_management,
            active_encoder: None,
            batch_ops: 0,
            batch_bytes: 0,
            shared_event,
            event_counter: 0,
            max_ops_per_batch,
            max_mb_per_batch,
            submitted_ops: 0,
            max_submitted_ops,
            last_command_buffer: None,
            profiling: None,
            timestamps: TimestampProfiler::default(),
            fault: Arc::new(FaultState::default()),
            batch_seq: 0,
            unconfirmed_writes: Vec::new(),
            batch_events: std::collections::VecDeque::new(),
            accounting: Arc::new(Accounting::default()),
            batch_estimate: 0.0,
            label: None,
            tag: None,
            max_batch_gpu_micros: {
                use cubecl_server::config::RuntimeConfig;
                cubecl_server::config::CubeClRuntimeConfig::get()
                    .streaming
                    .max_batch_gpu_micros
            },
        })
    }

    fn flush(stream: &mut Self::Stream, _failures: &mut ErrorGraph) -> Self::Event {
        use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLEvent};

        stream.event_counter += 1;
        let signal_value = stream.event_counter;

        let signal = Some((stream.shared_event.clone(), signal_value));

        let command_buffer = if let Some(active) = stream.active_encoder.take() {
            (*active.encoder).endEncoding();
            stream.account_batch(&active);

            // Metal never signals a faulted command buffer's event; its
            // completion handler does, after recording the fault. An injected
            // fault has to look the same, or a wait could see the signal
            // before the fault.
            if !active.inject_fault {
                let event_ref: &ProtocolObject<dyn MTLEvent> =
                    ProtocolObject::from_ref(&*stream.shared_event);
                (*active.command_buffer).encodeSignalEvent_value(event_ref, signal_value);
            }

            if !active.inject_fault {
                stream.batch_events.push_back((active.seq, signal_value));
                if stream.batch_events.len() > 512 {
                    stream.batch_events.pop_front();
                }
            }
            install_completion_handler(
                &active.command_buffer,
                active.temporaries,
                signal,
                stream.fault.clone(),
                active.seq,
                active.inject_fault,
            );
            (*active.command_buffer).commit();
            active.command_buffer
        } else {
            let signal_buffer = (*stream.queue)
                .commandBuffer()
                .expect("Failed to create command buffer");

            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*stream.shared_event);
            (*signal_buffer).encodeSignalEvent_value(event_ref, signal_value);
            // Carries no work, so it confirms nothing.
            install_completion_handler(
                &signal_buffer,
                Vec::new(),
                signal,
                stream.fault.clone(),
                0,
                false,
            );
            (*signal_buffer).commit();
            signal_buffer
        };

        let ops_in_batch = stream.batch_ops;

        // While profiling, collect command buffers that actually carried dispatches; skip
        // empty signal-only buffers (ops_in_batch == 0) so they don't widen the measured span.
        if ops_in_batch > 0
            && let Some(buffers) = stream.profiling.as_mut()
        {
            buffers.push(command_buffer.clone());
        }

        stream.last_command_buffer = Some(command_buffer);

        stream.batch_ops = 0;
        stream.batch_bytes = 0;

        stream.regulate(ops_in_batch);

        MetalEvent::new(
            stream.shared_event.clone(),
            signal_value,
            stream.fault.clone(),
        )
    }

    fn handle_cursor(stream: &Self::Stream, handle: &BufferBinding) -> u64 {
        // The slice cursor the sync logic compares against the origin stream's `last_synced`
        // to decide whether to wait. A freed/reallocated slice falls back to `u64::MAX`,
        // which conservatively forces a wait.
        stream
            .memory_management
            .get_cursor(handle.memory.clone())
            .unwrap_or(u64::MAX)
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) -> Result<(), ServerError> {
        event.wait_async(stream);
        Ok(())
    }

    fn wait_event_sync(event: Self::Event) -> Result<(), ServerError> {
        event.wait_sync()
    }
}

#[cfg(test)]
mod tests {
    use super::{Access, Hazards};

    fn at(buffer: usize, start: u64, end: u64) -> Access {
        Access {
            buffer,
            range: start..end,
        }
    }

    #[test]
    fn disjoint_dispatches_need_no_barrier() {
        let mut hazards = Hazards::default();
        assert!(!hazards.record(&[at(1, 0, 64)], &[at(1, 64, 128)]));
        // Reading what another dispatch reads, and writing next to what it writes.
        assert!(!hazards.record(&[at(1, 0, 64)], &[at(1, 128, 256)]));
        // The same bytes of another buffer.
        assert!(!hazards.record(&[], &[at(2, 64, 128)]));
    }

    #[test]
    fn every_overlap_kind_needs_a_barrier() {
        // Read after write.
        let mut hazards = Hazards::default();
        hazards.record(&[], &[at(1, 0, 64)]);
        assert!(hazards.record(&[at(1, 32, 96)], &[]));
        // Write after read.
        let mut hazards = Hazards::default();
        hazards.record(&[at(1, 0, 64)], &[]);
        assert!(hazards.record(&[], &[at(1, 63, 64)]));
        // Write after write.
        let mut hazards = Hazards::default();
        hazards.record(&[], &[at(1, 0, 64)]);
        assert!(hazards.record(&[], &[at(1, 0, 64)]));
    }

    #[test]
    fn a_barrier_forgets_what_came_before_it() {
        let mut hazards = Hazards::default();
        hazards.record(&[], &[at(1, 0, 64)]);
        assert!(hazards.record(&[at(1, 0, 64)], &[at(2, 0, 64)]));
        // Behind the barrier now: only the second dispatch's accesses count.
        assert!(!hazards.record(&[at(1, 0, 64)], &[at(3, 0, 64)]));
        assert!(hazards.record(&[at(2, 0, 8)], &[]));
    }
}
