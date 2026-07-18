//! Statistical test harness for RNGs.
//! This module provides utilities for testing RNGs, including:
//! * **Chi-squared test**: [`chisq`] module for assessing uniformity of RNG outputs.
//! * **Monte Carlo estimation of π**: [`mcpi`] module for estimating π using random sampling.
//! * **Serial test**: [`serial`] module, a multi-dimensional chi-squared test over tuples of consecutive outputs.
//! * **Runs test**: [`runs`] module for detecting sequential correlation via monotonic run lengths.
//! * **Kolmogorov-Smirnov test**: [`ks`] module for comparing the empirical CDF against the ideal uniform CDF.
//! * **Birthday spacing test**: [`birthday`] module (Marsaglia's DIEHARD test) for detecting short periods and lattice structure.
//! * **NIST SP 800-22 suite**: [`nist`] module, a bit-level subset (Frequency, Block Frequency, Runs, Longest Run of Ones, Cusum).
//! * **Paranoid meta-test**: [`paranoid`] module for elevating any single test into a battery-level test per NIST SP 800-22 §4.2.
//!
//! The test math, `Config`/`Result`/`Error`/`Verdict` types, and closure-based
//! engines all live in the [`cribler`] crate; this module supplies thin
//! `Rng32`/`Rng64`-typed wrappers over it.

pub mod birthday;
pub mod chisq;
pub mod ks;
pub mod mcpi;
pub mod nist;
pub mod paranoid;
#[cfg(feature = "rand")]
pub mod rand_adapter;
pub mod runs;
pub mod serial;
pub mod test;

pub use birthday::*;
pub use chisq::*;
pub use ks::*;
pub use mcpi::*;
pub use nist::*;
pub use paranoid::*;
#[cfg(feature = "rand")]
pub use rand_adapter::*;
pub use runs::*;
pub use serial::*;
pub use test::*;
