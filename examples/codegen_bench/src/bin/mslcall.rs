//! Does Metal keep a helper called at many sites as a function, and what does
//! each choice cost? `mslcall SITES BODY`: one helper of BODY transcendental
//! steps, called at SITES sites in a runtime loop, emitted four ways: a plain
//! function, `noinline`, `always_inline`, and pasted at every site by hand (what
//! CubeCL emits today). Prints compile time, the pipeline's thread limit (a
//! register-pressure proxy) and GPU time.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::fmt::Write;
use std::time::Instant;

fn body(steps: usize, x: &str, y: &str) -> String {
    let mut s = format!("float s = {x}; float t = {y};\n");
    for i in 0..steps {
        let c = 1.0 + i as f32 * 0.013;
        match i % 3 {
            0 => writeln!(
                s,
                "s = s * 0.5f + {c:.3}f * exp(-fabs(t) * 0.01f - fabs(s) * 0.001f);"
            )
            .unwrap(),
            1 => writeln!(s, "t = pow(fabs(s) + 1.0f, -0.2f) * {c:.3}f + t * 0.5f;").unwrap(),
            _ => writeln!(s, "s = (t > s) ? s + log(fabs(t) + 1.0f) : s - 0.3f * t;").unwrap(),
        }
    }
    s.push_str("return s + t;\n");
    s
}

fn source(variant: &str, sites: usize, steps: usize, salt: u128) -> String {
    let mut src = format!(
        "#include <metal_stdlib>\nusing namespace metal;\nconstant float SALT = {salt}.0e-40f;\n"
    );
    let attr = match variant {
        "plain" => "",
        "noinline" => "__attribute__((noinline))",
        "always" => "__attribute__((always_inline)) inline",
        _ => "",
    };
    if variant != "pasted" {
        writeln!(
            src,
            "{attr} float helper(float x, float y) {{\n{}}}",
            body(steps, "x", "y")
        )
        .unwrap();
    }
    src.push_str(
        "kernel void k(device float* out [[buffer(0)]], constant uint& n [[buffer(1)]], \
         uint id [[thread_position_in_grid]]) {\n float acc = float(id) * 1e-3f;\n \
         for (uint d = 0; d < n; d++) {\n  float fd = float(d);\n",
    );
    for i in 0..sites {
        let y = format!("fd * SALT + {}.0f", i);
        if variant == "pasted" {
            writeln!(
                src,
                "  {{ {} }}",
                body(steps, "acc", &y).replace("return s + t;", "acc = s + t;")
            )
            .unwrap();
        } else {
            writeln!(src, "  acc = helper(acc, {y});").unwrap();
        }
    }
    writeln!(src, " }}\n out[id] = acc + SALT;\n}}").unwrap();
    src
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let (sites, steps) = (
        args.first().copied().unwrap_or(64),
        args.get(1).copied().unwrap_or(60),
    );
    let device = MTLCreateSystemDefaultDevice().expect("should have a Metal device");
    let queue = device.newCommandQueue().unwrap();
    let threads = 1usize << 16;
    let out = device
        .newBufferWithLength_options(threads * 4, MTLResourceOptions::StorageModeShared)
        .unwrap();
    let n: u32 = 64;
    let n_buf = unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::from(&n).cast(),
            4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .unwrap();

    for variant in ["plain", "noinline", "always", "pasted"] {
        let mut compile = f64::MAX;
        let mut pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>> = None;
        let mut bytes = 0;
        for _ in 0..3 {
            let salt = Instant::now().elapsed().as_nanos()
                + std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
            let src = source(variant, sites, steps, salt);
            bytes = src.len();
            let start = Instant::now();
            let library = device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&src),
                    Some(&MTLCompileOptions::new()),
                )
                .unwrap_or_else(|e| panic!("{variant}: {:?}", e.localizedDescription()));
            let function = library
                .newFunctionWithName(&NSString::from_str("k"))
                .unwrap();
            let p = device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap();
            compile = compile.min(start.elapsed().as_secs_f64() * 1e3);
            pipeline = Some(p);
        }
        let pipeline = pipeline.unwrap();
        let mut gpu = Vec::new();
        for _ in 0..7 {
            let cb = queue.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(&pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&n_buf), 0, 1);
            }
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: threads,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            gpu.push((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3);
        }
        gpu.sort_by(f64::total_cmp);
        let checksum: f64 =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, threads) }
                .iter()
                .map(|&x| x as f64)
                .sum();
        println!(
            "{variant:>9}: src {:>7} B  compile {compile:7.1} ms  max threads {:>4}  gpu {:7.2} ms  checksum {checksum:.4e}",
            bytes,
            pipeline.maxTotalThreadsPerThreadgroup(),
            gpu[3]
        );
    }

    // Raw Metal round trip for an empty kernel: commit, wait, repeat.
    let src = "#include <metal_stdlib>\nkernel void e(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] += 1.0f; }";
    let lib = device
        .newLibraryWithSource_options_error(&NSString::from_str(src), None)
        .unwrap();
    let f = lib.newFunctionWithName(&NSString::from_str("e")).unwrap();
    let p = device
        .newComputePipelineStateWithFunction_error(&f)
        .unwrap();
    let mut rt = Vec::new();
    for _ in 0..200 {
        let start = Instant::now();
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&p);
        unsafe { enc.setBuffer_offset_atIndex(Some(&out), 0, 0) };
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        rt.push(start.elapsed().as_secs_f64() * 1e3);
    }
    rt.sort_by(f64::total_cmp);
    println!(
        "raw Metal commit+wait round trip: median {:.3} ms, p10 {:.3} ms",
        rt[100], rt[20]
    );
}
