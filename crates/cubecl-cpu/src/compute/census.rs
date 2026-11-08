//! With `CUBECL_CPU_CENSUS=<directory>` set, the JIT also writes every kernel it compiles there,
//! as `<key>.ll` (LLVM IR without a target, to be compiled ahead of time for each target) and
//! `<key>.kernel` (its name, whether calls run whole cubes, and its buffer access modes).

use std::path::PathBuf;

use cubecl_llvm::{CpuIr, PlironCompiler, PlironOptions};
use cubecl_server::kernel::{BufferIOAttr, KernelDefinition};

/// The census directory, when one is being taken.
pub(crate) fn directory() -> Option<PathBuf> {
    std::env::var_os("CUBECL_CPU_CENSUS").map(PathBuf::from)
}

/// Records the kernel `definition` under `key` in `directory`, once.
pub(crate) fn record(
    directory: &std::path::Path,
    key: &str,
    definition: KernelDefinition,
    options: &PlironOptions,
) {
    let ir_path = directory.join(format!("{key}.ll"));
    if ir_path.exists() {
        return;
    }
    let CpuIr {
        name,
        llvm,
        requirements,
        io,
    } = PlironCompiler::cpu_ir(definition, options)
        .unwrap_or_else(|error| panic!("the census could not lower a kernel: {error:?}"));
    let io: Vec<&str> = io
        .iter()
        .map(|mode| match mode {
            BufferIOAttr::ReadOnly => "ReadOnly",
            BufferIOAttr::WriteOnly => "WriteOnly",
            BufferIOAttr::ReadWrite => "ReadWrite",
            BufferIOAttr::Dead => "Dead",
        })
        .collect();
    let meta = format!(
        "name={name}\nwhole_cubes={}\nio={}\n",
        requirements.whole_cubes,
        io.join(",")
    );
    std::fs::create_dir_all(directory).expect("the census directory");
    // The metadata last: a kernel is in the census once both are written.
    std::fs::write(&ir_path, llvm).expect("the census IR");
    std::fs::write(directory.join(format!("{key}.kernel")), meta).expect("the census entry");
}
