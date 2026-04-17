//! # Universal RNG
//!
//! A collection of random number generators.
//!
//! This crate provides implementations of various pseudo-random number generators (PRNGs),
//! including:
//!
//! * **Mersenne Twister**: [`rng32::Mt19937`] (32-bit), [`rng64::Mt1993764`] (64-bit).
//! * **SIMD-oriented Fast Mersenne Twister**: [`rng32::Sfmt19937`] (32-bit), [`rng64::Sfmt1993764`] (64-bit).
//! * **Permuted Congruential Generator**: [`rng32::Pcg32`] (32-bit output).
//! * **Philox**: [`rng32::Philox32x4`] (4x32-bit), [`rng64::Philox64`] (2x64-bit).
//! * **Twisted Generalized Feedback Shift Register**: [`rng64::TwistedGFSR`] (64-bit).
//! * **Small Fast Chaotic**: [`rng64::Sfc64`] (64-bit).
//! * **Threefry**: [`rng32::Threefry32x4`] (4x32-bit), [`rng32::Threefry32x2`] (2x32-bit).
//! * **Threefish**: [`rng64::Threefish256`] (256-bit block cipher PRNG).
//! * **Xorshift**: [`rng32::Xorshift32`] (32-bit), [`rng64::Xorshift64`] (64-bit), [`rng128::Xorshift128`] (128-bit state).
//! * **Xorwow**: [`rng32::Xorwow`] (32-bit).
//! * **Xoshiro** (32-bit): [`rng32::Xoshiro128Pp`], [`rng32::Xoshiro128Ss`].
//! * **Xoshiro** (64-bit): [`rng64::Xoshiro256Pp`], [`rng64::Xoshiro256Ss`].
//! * **Xoroshiro**: [`rng64::xoroshiro::Xoroshiro128Pp`], [`rng64::xoroshiro::Xoroshiro128Ss`] (64-bit).
//! * **Linear Congruential Generator**: [`rng32::Lcg32`] (32-bit), [`rng64::Lcg64`] (64-bit).
//! * **Cet**: [`rng64::Cet64`] (64-bit), [`rng64::Cet256`] (256-bit state).
//! * **SplitMix**: [`rng32::SplitMix32`] (32-bit), [`rng64::SplitMix64`] (64-bit).
//! * **Jenkins Small Fast**: [`rng32::Jsf32`] (32-bit).
//!
//! Each generator supports generating uniform random numbers for various types (u32, u64, f32, f64)
//! and ranges.

/// A 32/64-bit random number generator trait.
pub mod rng;

/// Consolidated 32-bit random number generators.
pub mod rng32;

pub mod cabi32;

/// Consolidated 64-bit random number generators.
pub mod rng64;

pub mod cabi64;

pub(crate) mod _internal;

// #[cfg(feature = "wide")]
/// Wide SIMD-accelerated random number generators (e.g., `Sfmt19937x8`).
pub(crate) mod wide;

#[cfg(feature = "sampler")]
/// Weighted random selection traits (`Sampler32`, `Sampler64`).
pub mod sampler;

#[cfg(feature = "sampler")]
/// Weighted random selection for 32-bit RNGs (`Bst32`, `Alias32`).
pub mod sampler32;

#[cfg(feature = "sampler")]
/// Weighted random selection for 64-bit RNGs (`Bst64`, `Alias64`).
pub mod sampler64;

#[cfg(feature = "seedgen")]
pub mod seedgen;

#[cfg(feature = "testing")]
pub mod testing;

#[macro_use]
pub mod macros;

pub mod prelude {
    pub use crate::rng::{Rng32, Rng64};
    pub use crate::rng32::{
        Mt19937, Pcg32, Philox32x4, Sfc32, Sfmt19937, SplitMix32, Squares32, Squares32Simd,
        Squares32x8, Threefry32x2, Threefry32x4, Xorshift32, Xorshift128, Xorwow, Xoshiro128Pp,
    };
    pub use crate::rng64::{
        Cet64, Cet256, Mt1993764, Philox64, Sfc64, Sfmt1993764, SplitMix64, Threefish256,
        TwistedGFSR, Xoroshiro128Pp, Xoroshiro128Ss, Xoshiro256Pp, Xoshiro256Ss,
    };
}
