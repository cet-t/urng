//! # RNG Pack
//!
//! A collection of random number generators.
//!
//! This crate provides implementations of various pseudo-random number generators (PRNGs),
//! including:
//!
//! * **Mersenne Twister**: [`rng32::Mt19937`] (32-bit), [`rng64::Mt1993764`] (64-bit).
//! * **Permuted Congruential Generator**: [`rng32::Pcg32`] (32-bit output).
//! * **Philox**: [`rng32::Philox32`] (4x32-bit), [`rng64::Philox64`] (2x64-bit).
//! * **Twisted Generalized Feedback Shift Register**: [`rng64::TwistedGFSR`] (64-bit).
//! * **Small Fast Chaotic**: [`rng64::Sfc64`] (64-bit).
//! * **Xorshift**: [`rng32::Xorshift32`] (32-bit), [`rng64::Xorshift64`] (64-bit), [`rng128::Xorshift128`] (128-bit state).
//! * **Xorwow**: [`rng32::Xorwow`] (32-bit).
//! * **Xoshiro**: [`rng64::Xoshiro256Pp`], [`rng64::Xoshiro256Ss`] (64-bit).
//! * **Linear Congruential Generator**: [`rng32::Lcg32`] (32-bit), [`rng64::Lcg64`] (64-bit).
//! * **Cellular Automata**: [`rng64::Cet64`] (64-bit).
//! * **SplitMix**: [`rng32::SplitMix32`] (32-bit), [`rng64::SplitMix64`] (64-bit).
//!
//! ## Macros
//!
//! * **Random Generation**: [`next!`], [`rand!`]
//! * **Utilities**: [`wrap!`], [`search!`], [`choice!`]
//!
//! Each generator supports generating uniform random numbers for various types (u32, u64, f32, f64)
//! and ranges.
//!
//! ## C API
//!
//! This crate also exports C-compatible functions for creating, using, and freeing these generators,
//! allowing them to be used from other languages.

/// A 32/64-bit random number generator trait.
pub mod rng;

/// Consolidated 32-bit random number generators.
pub mod rng32;

/// Consolidated 64-bit random number generators.
pub mod rng64;

/// Consolidated 128-bit random number generators.
pub mod rng128;

/// A weighted random selection structure using a Binary Search Tree (BST) approach.
pub mod bst;

pub mod macros;
