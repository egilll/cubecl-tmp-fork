//! Kernels compiled ahead of time and linked into the binary, so a device needs no compiler.
//!
//! A kernel is found by [`kernel_key`]: a hash of its traced IR and the settings it was compiled
//! with, the same at build time and on every target. Once kernels are [`register`]ed, the CPU
//! runtime runs only those, and a kernel missing from them is an error, never a compilation.

use std::sync::{Arc, OnceLock};

use cubecl_common::hash::StableHasher;
pub use cubecl_server::kernel::BufferIOAttr;
use cubecl_server::kernel::KernelDefinition;

use super::executable::{Executable, KernelFn};

/// A kernel compiled ahead of time.
pub struct PrecompiledKernel {
    /// Its [`kernel_key`], in hex.
    pub key: &'static str,
    /// Its name, for messages.
    pub name: &'static str,
    /// Its entry point.
    pub entry: KernelFn,
    /// Whether each call must run whole cubes.
    pub whole_cubes: bool,
    /// Buffer access modes in binding order.
    pub io: &'static [BufferIOAttr],
}

static PRECOMPILED: OnceLock<&'static [PrecompiledKernel]> = OnceLock::new();

/// Runs `kernels` from now on, and nothing else. Call it before the first CPU device is opened.
pub fn register(kernels: &'static [PrecompiledKernel]) {
    let _ = PRECOMPILED.set(kernels);
}

/// The kernels registered, if any.
pub(crate) fn registered() -> Option<&'static [PrecompiledKernel]> {
    PRECOMPILED.get().copied()
}

/// The kernel with `key` among the registered ones.
pub(crate) fn find(key: &str) -> Option<Executable> {
    let kernel = registered()?.iter().find(|kernel| kernel.key == key)?;
    Some(Executable::new(
        kernel.entry,
        kernel.whole_cubes,
        Arc::from(kernel.io),
        None,
    ))
}

/// What identifies a kernel across builds and targets: its traced IR, its settings and how it is
/// compiled, hashed from their text, so that no field's width depends on the target.
pub fn kernel_key(definition: &KernelDefinition, compilation: &str) -> String {
    let text = key_text(definition, compilation);
    format!("{:032x}", StableHasher::const_hash(text.as_bytes()))
}

/// What [`kernel_key`] hashes. The kernel's generated name is left out: its suffix hashes a type
/// identity, which differs between targets.
fn key_text(definition: &KernelDefinition, compilation: &str) -> String {
    format!(
        "{}\n{:?}\n{:?}\n{compilation}",
        definition.body, definition.info, definition.settings
    )
    .replace(&definition.settings.kernel_name, "kernel")
}
