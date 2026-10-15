//! The build identity of the code that compiled a kernel.
//!
//! Compiled kernels are cached across runs, keyed by the build that produced
//! them, so a rebuild never serves another build's binaries. Kernels may be
//! defined and instantiated in shared libraries as well as in the
//! executable, so on Apple platforms the identity covers every image the
//! process has loaded: a rebuild of any of them changes it. `buildid` reads
//! only the executable's header there, a symbol a shared library holding
//! CubeCL could not even link against.

/// The build identifier of the running code, when the platform records
/// one: on Apple platforms the `LC_UUID` of every loaded Mach-O image, in
/// load order; elsewhere whatever `buildid` finds.
pub fn build_id() -> Option<&'static [u8]> {
    imp::build_id()
}

#[cfg(target_vendor = "apple")]
mod imp {
    use std::sync::OnceLock;
    use std::vec::Vec;

    /// `MH_MAGIC_64`: a 64-bit Mach-O header.
    const MAGIC_64: u32 = 0xfeed_facf;
    /// The `LC_UUID` load command.
    const LC_UUID: u32 = 0x1b;
    /// The size of `mach_header_64`.
    const HEADER: usize = 32;

    unsafe extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_header(image: u32) -> *const u8;
    }

    pub fn build_id() -> Option<&'static [u8]> {
        static IDENTITY: OnceLock<Vec<u8>> = OnceLock::new();
        let identity = IDENTITY.get_or_init(|| {
            // SAFETY: dyld's image list may be read at any time.
            let count = unsafe { _dyld_image_count() };
            (0..count)
                // SAFETY: every index below the count names a loaded image.
                .filter_map(|image| uuid(unsafe { _dyld_get_image_header(image) }))
                .flatten()
                .collect()
        });
        (!identity.is_empty()).then_some(identity.as_slice())
    }

    /// The `LC_UUID` of the Mach-O image whose header is at `base`.
    fn uuid(base: *const u8) -> Option<[u8; 16]> {
        if base.is_null() {
            return None;
        }
        // SAFETY: `base` is the mapped header of a loaded image, which stays
        // mapped while it is read; its load commands follow the header.
        unsafe {
            let read = |at: usize| base.add(at).cast::<u32>().read_unaligned();
            if read(0) != MAGIC_64 {
                return None;
            }
            let (commands, mut at) = (read(16), HEADER);
            for _ in 0..commands {
                let (command, size) = (read(at), read(at + 4) as usize);
                if command == LC_UUID {
                    return Some(base.add(at + 8).cast::<[u8; 16]>().read_unaligned());
                }
                if size == 0 {
                    return None;
                }
                at += size;
            }
        }
        None
    }
}

#[cfg(not(target_vendor = "apple"))]
mod imp {
    pub fn build_id() -> Option<&'static [u8]> {
        buildid::build_id()
    }
}
