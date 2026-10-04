//! The gate probe for separate compute queues: the round trip of a small,
//! frame-like command buffer while long compute dispatches keep the GPU busy,
//! with the compute on its own queue versus on the same queue. Also counts
//! command buffers that fail (aborts). `gate [SECONDS] [DISPATCH_MS]`.
//!
//! Run it on a healthy GPU (see `waits`), ideally with a window animating so
//! the compositor competes too.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const SOURCE: &str = "#include <metal_stdlib>
using namespace metal;
kernel void spin(device uint* o [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    uint acc = i;
    for (uint k = 0; k < n; k++) { acc = acc * 1664525u + 1013904223u; }
    o[i] = acc;
}";

struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    spin: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    out: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

impl Gpu {
    fn encode(
        &self,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        threads: usize,
        n: u32,
    ) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&self.spin);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&self.out), 0, 0);
            enc.setBytes_length_atIndex(std::ptr::NonNull::from(&n).cast(), 4, 1);
        }
        let size = |w| MTLSize {
            width: w,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(size(threads), size(256));
        enc.endEncoding();
        cb.commit();
        cb
    }
}

fn percentile(v: &mut [Duration], p: f64) -> Duration {
    v.sort();
    v[((v.len() - 1) as f64 * p) as usize]
}

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<f64>().unwrap());
    let seconds = args.next().unwrap_or(2.0);
    let dispatch_ms = args.next().unwrap_or(3.0);

    let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
        .unwrap();
    let spin = device
        .newComputePipelineStateWithFunction_error(
            &library
                .newFunctionWithName(&NSString::from_str("spin"))
                .unwrap(),
        )
        .unwrap();
    let out = device
        .newBufferWithLength_options(4 << 20, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let gpu = Arc::new(Gpu {
        device: device.clone(),
        spin,
        out,
    });

    // Calibrate the compute dispatch to about `dispatch_ms`.
    let queue = device.newCommandQueue().unwrap();
    let threads = 1 << 20;
    let mut n = 1000u32;
    loop {
        let cb = gpu.encode(&queue, threads, n);
        cb.waitUntilCompleted();
        let ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
        if ms >= dispatch_ms * 0.8 || n > 1 << 24 {
            n = ((n as f64) * dispatch_ms / ms.max(0.01)) as u32;
            break;
        }
        n *= 2;
    }

    for separate in [true, false] {
        let compute_queue = gpu.device.newCommandQueue().unwrap();
        let frame_queue = match separate {
            true => gpu.device.newCommandQueue().unwrap(),
            false => compute_queue.clone(),
        };
        let stop = Arc::new(AtomicBool::new(false));
        let aborts = Arc::new(AtomicU64::new(0));
        let compute = {
            let (gpu, stop, aborts, queue) = (
                gpu.clone(),
                stop.clone(),
                aborts.clone(),
                compute_queue.clone(),
            );
            let queue = SendQueue(queue);
            std::thread::spawn(move || {
                let queue = queue;
                // Keep about three dispatches in flight.
                let mut in_flight = std::collections::VecDeque::new();
                while !stop.load(Ordering::Relaxed) {
                    in_flight.push_back(gpu.encode(&queue.0, threads, n));
                    if in_flight.len() > 3 {
                        let cb: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                            in_flight.pop_front().unwrap();
                        cb.waitUntilCompleted();
                        if cb.status() == MTLCommandBufferStatus::Error {
                            aborts.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                for cb in in_flight {
                    cb.waitUntilCompleted();
                }
            })
        };
        let mut frames = Vec::new();
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < seconds {
            let t = Instant::now();
            let cb = gpu.encode(&frame_queue, 4096, 64);
            cb.waitUntilCompleted();
            if cb.status() == MTLCommandBufferStatus::Error {
                aborts.fetch_add(1, Ordering::Relaxed);
            }
            frames.push(t.elapsed());
            std::thread::sleep(Duration::from_millis(8));
        }
        stop.store(true, Ordering::Relaxed);
        compute.join().unwrap();
        println!(
            "{:>14}: frame round trip p50 {:?} p90 {:?} max {:?} ({} frames), aborts {}",
            if separate {
                "separate queue"
            } else {
                "same queue"
            },
            percentile(&mut frames, 0.5),
            percentile(&mut frames, 0.9),
            percentile(&mut frames, 1.0),
            frames.len(),
            aborts.load(Ordering::Relaxed)
        );
    }
}

struct SendQueue(Retained<ProtocolObject<dyn MTLCommandQueue>>);
unsafe impl Send for SendQueue {}
