#[doc = include_str!("../README.md")]
/// A 32/64-bit random number generator trait.
pub mod rng;
pub use crate::rng::*;

#[cfg(feature = "simd")]
pub mod rngv;
#[cfg(feature = "simd")]
pub use crate::rngv::*;

/// Consolidated 32-bit random number generators.
pub mod rng32;
#[allow(ambiguous_glob_reexports)]
pub use crate::rng32::*;

#[cfg(feature = "rand")]
pub mod rand32;

#[cfg(feature = "cabi")]
pub mod cabi32;
#[cfg(feature = "cabi")]
pub use crate::cabi32::*;

/// Consolidated 64-bit random number generators.
pub mod rng64;
#[allow(ambiguous_glob_reexports)]
pub use crate::rng64::*;

#[cfg(feature = "rand")]
pub(crate) mod rand64;

#[cfg(feature = "cabi")]
pub mod cabi64;
#[cfg(feature = "cabi")]
pub use crate::cabi64::*;

pub mod shuffle;
pub use crate::shuffle::*;

pub mod choice;

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
