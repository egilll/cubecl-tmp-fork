//! Raw Metal: what does waiting for one tiny dispatch cost, by how the CPU waits? Median of 200
//! round trips (commit, then wait) for each method.
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary,
    MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use std::time::{Duration, Instant};

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("should have a Metal device");
    let queues: Vec<_> = (0..4).map(|_| device.newCommandQueue().unwrap()).collect();
    let src = "#include <metal_stdlib>\nkernel void e(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] += 1.0f; }";
    let lib = device
        .newLibraryWithSource_options_error(&NSString::from_str(src), None)
        .unwrap();
    let f = lib.newFunctionWithName(&NSString::from_str("e")).unwrap();
    let p = device.newComputePipelineStateWithFunction_error(&f).unwrap();
    let out = device
        .newBufferWithLength_options(4096, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let event = device.newSharedEvent().unwrap();
    let probe_lib = device
        .newLibraryWithSource_options_error(&NSString::from_str("kernel void probe() {}"), None)
        .unwrap();
    let probe = device
        .newComputePipelineStateWithFunction_error(
            &probe_lib.newFunctionWithName(&NSString::from_str("probe")).unwrap(),
        )
        .unwrap();
    let mut value = 0u64;

    let mut round = |queue: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandQueue>>, method: &str| -> Duration {
        let start = Instant::now();
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&p);
        unsafe { enc.setBuffer_offset_atIndex(Some(&out), 0, 0) };
        let size = |w| MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(size(1024), size(256));
        enc.endEncoding();
        value += 1;
        let event_ref: &objc2::runtime::ProtocolObject<dyn MTLEvent> =
            objc2::runtime::ProtocolObject::from_ref(&*event);
        cb.encodeSignalEvent_value(event_ref, value);
        cb.commit();
        match method {
            "waitUntilCompleted" => cb.waitUntilCompleted(),
            "event wait" => {
                event.waitUntilSignaledValue_timeoutMS(value, 10_000);
            }
            "spin status" => while cb.status() != MTLCommandBufferStatus::Completed {
                std::hint::spin_loop()
            },
            "spin event" => while event.signaledValue() < value {
                std::hint::spin_loop()
            },
            _ => unreachable!(),
        }
        start.elapsed()
    };
    for (q, queue) in queues.iter().enumerate() {
        let mut t: Vec<_> = (0..50).map(|_| round(queue, "waitUntilCompleted")).collect();
        t.sort();
        // Pipelined: 100 command buffers back to back, then wait for the last.
        let start = Instant::now();
        let mut last = None;
        for _ in 0..100 {
            let cb = queue.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&p);
            unsafe { enc.setBuffer_offset_atIndex(Some(&out), 0, 0) };
            let size = |w| MTLSize { width: w, height: 1, depth: 1 };
            enc.dispatchThreads_threadsPerThreadgroup(size(1024), size(256));
            enc.endEncoding();
            cb.commit();
            last = Some(cb);
        }
        last.unwrap().waitUntilCompleted();
        println!("queue {q}: round trip {:?}, 100 pipelined {:?}", t[25], start.elapsed());
    }
    drop(queues);
    // Recreated after the first four are gone.
    for q in 0..4 {
        let queue = device.newCommandQueue().unwrap();
        let mut t: Vec<_> = (0..50).map(|_| round(&queue, "waitUntilCompleted")).collect();
        t.sort();
        let empty: Vec<_> = (0..3)
            .map(|_| {
                let start = Instant::now();
                let cb = queue.commandBuffer().unwrap();
                cb.commit();
                cb.waitUntilCompleted();
                start.elapsed()
            })
            .collect();
        let blit: Vec<_> = (0..3)
            .map(|_| {
                use objc2_metal::MTLBlitCommandEncoder;
                let start = Instant::now();
                let cb = queue.commandBuffer().unwrap();
                let enc = cb.blitCommandEncoder().unwrap();
                enc.fillBuffer_range_value(&out, objc2_foundation::NSRange::new(0, 4), 0);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                start.elapsed()
            })
            .collect();
        let encoder: Vec<_> = (0..3)
            .map(|_| {
                let start = Instant::now();
                let cb = queue.commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                enc.setComputePipelineState(&probe);
                let one = MTLSize { width: 1, height: 1, depth: 1 };
                enc.dispatchThreadgroups_threadsPerThreadgroup(one, one);
                enc.endEncoding();
                cb.commit();
                cb.waitUntilCompleted();
                start.elapsed()
            })
            .collect();
        println!("new queue {q}: round trip {:?}, empty {empty:?}, blit {blit:?}, empty kernel {encoder:?}", t[25]);
    }
}
