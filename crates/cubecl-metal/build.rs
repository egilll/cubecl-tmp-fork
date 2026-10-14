//! Fingerprints the sources that lower a kernel's IR to MSL. Cached MSL is keyed by the kernel's
//! IR and this fingerprint rather than by the binary, so it survives rebuilds of the program that
//! leave both unchanged.

use std::{
    hash::{DefaultHasher, Hasher},
    path::Path,
};

/// The crates whose code takes part in tracing, optimizing or emitting a kernel's source.
const CODEGEN_CRATES: &[&str] = &[
    "cubecl-common",
    "cubecl-core",
    "cubecl-cpp",
    "cubecl-environment",
    "cubecl-ir",
    "cubecl-macros",
    "cubecl-macros-internal",
    "cubecl-metal",
    "cubecl-opt",
    "cubecl-runtime",
    "cubecl-server",
    "cubecl-zspace",
];

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let crates = Path::new(&manifest_dir).parent().unwrap();

    let mut hasher = DefaultHasher::new();
    hasher.write(std::env::var("CARGO_PKG_VERSION").unwrap().as_bytes());
    for name in CODEGEN_CRATES {
        let dir = crates.join(name);
        // Absent outside a checkout of the repository, where the version stands in.
        if !dir.exists() {
            continue;
        }
        for entry in ["src", "Cargo.toml"] {
            let path = dir.join(entry);
            println!("cargo::rerun-if-changed={}", path.display());
            hash_tree(&mut hasher, crates, &path);
        }
    }

    println!(
        "cargo::rustc-env=CUBECL_METAL_CODEGEN_FINGERPRINT={:016x}",
        hasher.finish()
    );
}

fn hash_tree(hasher: &mut DefaultHasher, root: &Path, path: &Path) {
    if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        entries.sort();
        for entry in entries {
            hash_tree(hasher, root, &entry);
        }
    } else if let Ok(contents) = std::fs::read(path) {
        hasher.write(
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        hasher.write(&contents);
    }
}
