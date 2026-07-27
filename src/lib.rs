#![doc = include_str!("../README.md")]

/// A 32/64-bit random number generator trait.
pub mod rng;
pub use crate::rng::*;

#[cfg(feature = "simd")]
pub mod rngv;
#[cfg(feature = "simd")]
pub use crate::rngv::*;

/// Stateful pseudo-random number generators (32/64-bit).
pub mod prng;
#[allow(ambiguous_glob_reexports)]
pub use crate::prng::*;

/// Counter-based random number generators (32/64-bit).
pub mod cbrng;
#[allow(ambiguous_glob_reexports)]
pub use crate::cbrng::*;

#[cfg(feature = "rand")]
pub mod rand32;

#[cfg(feature = "cabi")]
pub mod cabi32;
#[cfg(feature = "cabi")]
pub use crate::cabi32::*;

#[cfg(feature = "rand")]
pub(crate) mod rand64;

#[cfg(feature = "cabi")]
pub mod cabi64;
#[cfg(feature = "cabi")]
pub use crate::cabi64::*;

pub mod shuffle;
pub use crate::shuffle::*;

pub mod choice;
pub use crate::choice::*;

pub(crate) mod _internal;

/// Wide SIMD-accelerated random number generators (e.g., `Sfc32x8`).
#[cfg(feature = "wide")]
pub mod wide;

/// Weighted random selection traits (`Sampler32`, `Sampler64`).
#[cfg(feature = "sampler")]
pub mod sampler;

/// Weighted random selection for 32-bit RNGs (`Bst32`, `Alias32`).
#[cfg(feature = "sampler")]
pub mod sampler32;

/// Weighted random selection for 64-bit RNGs (`Bst64`, `Alias64`).
#[cfg(feature = "sampler")]
pub mod sampler64;

#[cfg(feature = "seedgen")]
pub mod seedgen;
#[cfg(feature = "seedgen")]
pub use crate::seedgen::*;

#[macro_use]
pub mod macros;
