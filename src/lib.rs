//! # RNG Pack
//!
//! A collection of random number generators.
//!
//! This crate provides implementations of various pseudo-random number generators (PRNGs),
//! including:
//!
//! * **Mersenne Twister**: [`rng32::Mt19937`] (32-bit) and [`rng64::Mt1993764`] (64-bit).
//! * **PCG**: [`rng32::Pcg32`] (Permuted Congruential Generator).
//! * **Philox**: [`rng32::Philox32`] (4x32) and [`rng64::Philox64`] (2x64), counter-based RNGs.
//! * **Twisted GFSR**: [`rng32::TwistedGFSR`].
//! * **SFC**: [`rng64::Sfc64`] (64-bit).
//! * **Xorshift**: [`rng32::Xorshift32`], [`rng64::Xorshift64`], and [`rng128::Xorshift128`].
//!
//! Each generator supports generating uniform random numbers for various types (`u32`, `u64`, `f32`, `f64`)
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
