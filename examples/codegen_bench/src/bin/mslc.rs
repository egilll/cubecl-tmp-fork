//! Times Metal's compilation of MSL files: `mslc FILE...`, each file holding one kernel, compiled
//! with the default options wgpu uses for passthrough MSL.
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary};
use std::time::Instant;

fn main() {
    let device = MTLCreateSystemDefaultDevice().expect("should have a Metal device");
    for path in std::env::args().skip(1) {
        let source = std::fs::read_to_string(&path).unwrap();
        // Distinct text per run, so the driver's shader cache never answers.
        let mut best = f64::MAX;
        for run in 0..3 {
            let source = format!("{source}\n// run {run} {:?}\n", Instant::now());
            let options = MTLCompileOptions::new();
            let start = Instant::now();
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
                .unwrap_or_else(|err| panic!("{path}: {:?}", err.localizedDescription()));
            let name = library.functionNames().objectAtIndex(0);
            let function = library.newFunctionWithName(&name).unwrap();
            device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap();
            best = best.min(start.elapsed().as_secs_f64() * 1e3);
        }
        println!("{best:8.1} ms  {path}");
    }
}
