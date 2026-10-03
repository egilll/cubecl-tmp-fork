use core::fmt::Display;
use core::sync::atomic::{AtomicU64, Ordering};

/// What a device has been asked to do since it came up, as counts.
///
/// Counts are cheap enough to keep on every launch, and they answer the
/// question timings cannot: whether two versions of an application do the
/// same amount of work. A round trip to the GPU costs milliseconds on some
/// platforms whatever the kernel, so a workload that syncs more often is
/// slower before any kernel is.
///
/// Read with [`Client::activity`](crate::client::Client::activity), and
/// subtract two readings to count a region:
///
/// ```ignore
/// let before = client.activity();
/// run_round(&client);
/// println!("{}", client.activity() - before);
/// ```
#[derive(Debug, Default)]
pub struct ActivityCounters {
    launches: AtomicU64,
    syncs: AtomicU64,
    reads: AtomicU64,
    read_bytes: AtomicU64,
    compilations: AtomicU64,
    compile_micros: AtomicU64,
}

/// A reading of [`ActivityCounters`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Activity {
    /// Kernels launched, including the ones skipped for an empty grid.
    pub launches: u64,
    /// Calls to `sync` or `sync_buffers`.
    pub syncs: u64,
    /// Read requests, each a host round trip however many buffers it reads.
    pub reads: u64,
    /// Bytes requested by those reads.
    pub read_bytes: u64,
    /// Kernels compiled from IR, cache hits excluded.
    pub compilations: u64,
    /// Time spent defining and compiling those kernels in `CubeCL`, before
    /// the driver's own compiler, in microseconds.
    pub compile_micros: u64,
}

impl ActivityCounters {
    /// A kernel launch.
    pub fn launch(&self) {
        self.launches.fetch_add(1, Ordering::Relaxed);
    }

    /// A sync.
    pub fn sync(&self) {
        self.syncs.fetch_add(1, Ordering::Relaxed);
    }

    /// A read of `bytes` bytes.
    pub fn read(&self, bytes: u64) {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.read_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// A compilation that took `micros` microseconds in `CubeCL`.
    pub fn compilation(&self, micros: u64) {
        self.compilations.fetch_add(1, Ordering::Relaxed);
        self.compile_micros.fetch_add(micros, Ordering::Relaxed);
    }

    /// The counts so far.
    pub fn reading(&self) -> Activity {
        Activity {
            launches: self.launches.load(Ordering::Relaxed),
            syncs: self.syncs.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            compilations: self.compilations.load(Ordering::Relaxed),
            compile_micros: self.compile_micros.load(Ordering::Relaxed),
        }
    }
}

impl core::ops::Sub for Activity {
    type Output = Activity;

    fn sub(self, rhs: Self) -> Self::Output {
        Activity {
            launches: self.launches.saturating_sub(rhs.launches),
            syncs: self.syncs.saturating_sub(rhs.syncs),
            reads: self.reads.saturating_sub(rhs.reads),
            read_bytes: self.read_bytes.saturating_sub(rhs.read_bytes),
            compilations: self.compilations.saturating_sub(rhs.compilations),
            compile_micros: self.compile_micros.saturating_sub(rhs.compile_micros),
        }
    }
}

impl Display for Activity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} launches, {} syncs, {} reads ({} bytes), {} compilations ({:.1} ms)",
            self.launches,
            self.syncs,
            self.reads,
            self.read_bytes,
            self.compilations,
            self.compile_micros as f64 / 1000.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn a_region_is_the_difference_of_two_readings() {
        let counters = ActivityCounters::default();
        counters.launch();
        let before = counters.reading();
        counters.launch();
        counters.launch();
        counters.sync();
        counters.read(16);
        counters.compilation(1500);

        let region = counters.reading() - before;
        assert_eq!(
            region,
            Activity {
                launches: 2,
                syncs: 1,
                reads: 1,
                read_bytes: 16,
                compilations: 1,
                compile_micros: 1500,
            }
        );
        assert_eq!(
            region.to_string(),
            "2 launches, 1 syncs, 1 reads (16 bytes), 1 compilations (1.5 ms)"
        );
    }
}
