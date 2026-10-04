//! Does MSL text carry its own math semantics? Compiles one kernel with several source prefixes,
//! each under wgpu's passthrough options (`MTLCompileOptions::new()`, fast math) and under native
//! `cubecl-metal`'s (safe math, precise functions), runs it, and counts the outputs whose bits differ
//! from the reference (no prefix, safe + precise). `mathmode [PREFIX_FILE...]`.
use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion,
    MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode, MTLResourceOptions, MTLSize,
};

const FUNCS: &[(&str, &str)] = &[
    ("exp", "exp(x * 0.05f)"),
    ("log", "log(fabs(x) + 1e-3f)"),
    ("pow", "pow(fabs(x) + 0.5f, y * 0.1f)"),
    ("sin", "sin(x * 3.0f)"),
    ("cos", "cos(x * 3.0f)"),
    ("tanh", "tanh(x * 0.1f)"),
    ("sqrt", "sqrt(fabs(x))"),
    ("rsqrt", "rsqrt(fabs(x) + 1e-3f)"),
    ("div", "x / y"),
    ("contract", "x * y + z"),
    ("reassoc", "(x + 1e8f) - 1e8f"),
    ("isnan", "isnan(x / (y - y)) ? 1.0f : 0.0f"),
    ("inf", "isinf(x * 1e38f) ? 1.0f : 0.0f"),
    ("fast::exp", "fast::exp(x * 0.05f)"),
    ("precise::exp", "precise::exp(x * 0.05f)"),
];

fn kernel() -> String {
    let mut s = String::from(
        "kernel void k(device const float* a [[buffer(0)]], device float* o [[buffer(1)]], \
         uint i [[thread_position_in_grid]]) {\n float x = a[i]; float y = a[i ^ 1] + 0.25f; \
         float z = a[i ^ 2];\n",
    );
    let precise = std::env::var("PRECISE").is_ok();
    for (j, (_, expr)) in FUNCS.iter().enumerate() {
        let mut expr = expr.to_string();
        if precise && !expr.contains("::") {
            for f in ["exp", "log", "pow", "sin", "cos", "tanh", "rsqrt", "sqrt"] {
                expr = expr.replacen(&format!("{f}("), &format!("precise::{f}("), 1);
            }
            expr = expr.replace("rprecise::sqrt", "rsqrt");
        }
        s += &format!(" o[i * {} + {j}] = {expr};\n", FUNCS.len());
    }
    s + "}\n"
}

fn options(native: bool) -> Retained<MTLCompileOptions> {
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(MTLLanguageVersion::Version3_2);
    if native {
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
    }
    options
}

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("should have a Metal device");
    let queue = device.newCommandQueue().unwrap();
    let n = 1usize << 16;
    let input: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.7311).sin() * 100.0 + (i % 7) as f32 * 1e-3)
        .collect();
    let a = unsafe {
        device.newBufferWithBytes_length_options(
            std::ptr::NonNull::new(input.as_ptr() as *mut _).unwrap(),
            n * 4,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .unwrap();
    let o = device
        .newBufferWithLength_options(n * 4 * FUNCS.len(), MTLResourceOptions::StorageModeShared)
        .unwrap();

    let run = |source: &str, native: bool| -> Option<Vec<u32>> {
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options(native)))
            .map_err(|e| eprintln!("  compile error: {}", e.localizedDescription()))
            .ok()?;
        let f = library.newFunctionWithName(&NSString::from_str("k"))?;
        let p = device.newComputePipelineStateWithFunction_error(&f).ok()?;
        let cb = queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();
        enc.setComputePipelineState(&p);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&a), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&o), 0, 1);
        }
        let size = |w| MTLSize { width: w, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(size(n), size(256));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Some(
            unsafe {
                std::slice::from_raw_parts(o.contents().as_ptr() as *const u32, n * FUNCS.len())
            }
            .to_vec(),
        )
    };

    let header = "#include <metal_stdlib>\nusing namespace metal;\n";
    let reference = run(&format!("{header}{}", kernel()), true).unwrap();

    let idx = |name: &str| FUNCS.iter().position(|(f, _)| *f == name).unwrap();
    let (div, sqrt, exp) = (idx("div"), idx("sqrt"), idx("exp"));
    let mut wrong = [0; 3];
    for i in 0..n {
        let (x, y) = (input[i], input[i ^ 1] + 0.25);
        let at = |j: usize| f32::from_bits(reference[i * FUNCS.len() + j]);
        wrong[0] += (at(div) != x / y) as usize;
        wrong[1] += (at(sqrt) != x.abs().sqrt()) as usize;
        wrong[2] += (at(exp) != (x * 0.05).exp()) as usize;
    }
    println!("native vs CPU, inexact of {n}: div {} sqrt {} exp {}", wrong[0], wrong[1], wrong[2]);
    let mut prefixes = vec![("none".to_string(), String::new())];
    for path in std::env::args().skip(1) {
        prefixes.push((path.clone(), std::fs::read_to_string(&path).unwrap()));
    }
    for (name, prefix) in prefixes {
        for native in [false, true] {
            let source = format!("{prefix}{header}{}", kernel());
            let route = if native { "native" } else { "wgpu  " };
            let Some(out) = run(&source, native) else {
                println!("{name} / {route}: failed");
                continue;
            };
            let diffs: Vec<String> = FUNCS
                .iter()
                .enumerate()
                .filter_map(|(j, (fname, _))| {
                    let d = (0..n)
                        .filter(|i| {
                            out[i * FUNCS.len() + j] != reference[i * FUNCS.len() + j]
                        })
                        .count();
                    (d > 0).then(|| format!("{fname}:{d}"))
                })
                .collect();
            println!("{name} / {route}: {}", if diffs.is_empty() { "same".into() } else { diffs.join(" ") });
        }
    }
}
