#![doc = include_str!("../README.md")]
#![allow(ambiguous_glob_reexports)]

/// A 32/64-bit random number generator trait.
mod rng;
pub use crate::rng::*;

#[cfg(feature = "simd")]
mod rngv;
#[cfg(feature = "simd")]
pub use crate::rngv::*;

/// Stateful pseudo-random number generators (32/64-bit).
mod prng;
pub use crate::prng::*;

/// Counter-based random number generators (32/64-bit).
mod cbrng;
pub use crate::cbrng::*;

/// Cipher-based random number generators (32-bit).
mod crng;
pub use crate::crng::*;

#[cfg(feature = "rand")]
mod rand;

/// C-compatible ABI wrappers (32/64-bit).
#[cfg(feature = "cabi")]
pub mod cabi;
#[cfg(feature = "cabi")]
pub use crate::cabi::{cbrng::*, prng::*};

mod shuffle;
pub use crate::shuffle::*;

mod choice;
pub use crate::choice::*;

pub(crate) mod _internal;
pub(crate) use crate::_internal::*;

#[cfg(feature = "wide")]
pub mod wide;

#[cfg(feature = "sampler")]
mod sampler;
#[cfg(feature = "sampler")]
pub use crate::sampler::*;

#[cfg(feature = "seedgen")]
mod seedgen;
#[cfg(feature = "seedgen")]
pub use crate::seedgen::*;

#[macro_use]
mod macros;
