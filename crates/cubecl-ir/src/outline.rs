//! Bookkeeping for device functions traced once and called many times.
//!
//! A `#[cube(outline)]` function is traced into a private function of the
//! kernel module the first time it is called with a given specialization, and
//! called again for every later call with the same one. The specialization is
//! everything that can change the traced body: the function, its generic
//! types, its comptime arguments, the type of each runtime argument (or the
//! value of a constant one), the instruction modes in effect and the kernel's
//! type registrations.

use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

use pliron::{identifier::Identifier, r#type::TypeHandle};

use crate::{ElemType, InstructionModes, value::ConstantValue};

/// One runtime argument of an outlined call, as far as specialization goes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OutlineArgKey {
    /// A value only known when the kernel runs: the function takes it as a
    /// parameter of this type.
    Value(TypeHandle),
    /// A value known while tracing: the body is traced with it, as an inlined
    /// call would be, and does not take it as a parameter.
    Constant(ConstantBits, ElemType),
}

/// A [`ConstantValue`] compared and hashed by its bits, so a constant can be
/// part of a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConstantBits(u8, u64, u64);

impl From<ConstantValue> for ConstantBits {
    fn from(value: ConstantValue) -> Self {
        match value {
            ConstantValue::Int(v) => ConstantBits(0, v as u64, 0),
            ConstantValue::Float(v) => ConstantBits(1, v.to_bits(), 0),
            ConstantValue::UInt(v) => ConstantBits(2, v, 0),
            ConstantValue::Bool(v) => ConstantBits(3, v as u64, 0),
            ConstantValue::Complex(re, im) => ConstantBits(4, re.to_bits(), im.to_bits()),
        }
    }
}

/// Everything an outlined body depends on. Two calls with equal keys trace
/// the same body.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutlineKey {
    /// The function: its module path and name.
    pub function: &'static str,
    /// The names of its generic types, in declaration order.
    pub generics: Vec<&'static str>,
    /// A hash of its comptime arguments and const generics.
    pub comptime: u64,
    /// Its runtime arguments.
    pub args: Vec<OutlineArgKey>,
    /// Fast-math and other modes in effect at the call.
    pub modes: InstructionModes,
    /// A hash of what the arguments carry besides their runtime values, such
    /// as a struct's comptime fields.
    pub arg_comptime: u64,
    /// A hash of the kernel's type and size registrations.
    pub registrations: u64,
}

/// A function the cache holds.
#[derive(Debug, Clone)]
pub enum Outlined {
    /// Traced into this function, which every call with the same key calls.
    Function { symbol: Identifier, ty: TypeHandle },
    /// Traced inline at every call, for the reason given: the body could not
    /// stand on its own.
    Inline(OutlineFallback),
}

/// Why a body could not become a function of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlineFallback {
    /// It used values of its caller that are not its arguments.
    Captures,
    /// It terminates the kernel, which only the kernel's own body can do.
    Terminates,
}

/// The hasher for the parts of an [`OutlineKey`] that are hashes: stable
/// within a process, which is as long as a key lives.
pub fn outline_hasher() -> impl Hasher {
    foldhash::fast::FixedState::with_seed(0x6375_6265_636c).build_hasher()
}

/// A hash of a map's entries regardless of iteration order.
pub fn hash_unordered<K: Hash, V: Hash>(entries: impl Iterator<Item = (K, V)>) -> u64 {
    let mut hashes: Vec<u64> = entries
        .map(|entry| {
            let mut hasher = foldhash::fast::FixedState::with_seed(0).build_hasher();
            entry.hash(&mut hasher);
            hasher.finish()
        })
        .collect();
    hashes.sort_unstable();
    let mut hasher = foldhash::fast::FixedState::with_seed(0).build_hasher();
    hashes.hash(&mut hasher);
    hasher.finish()
}

use core::hash::BuildHasher;
