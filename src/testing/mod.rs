//! Statistical test harness for RNGs.
//! This module provides utilities for testing RNGs, including:
//! * **Chi-squared test**: [`chisq`] module for assessing uniformity of RNG outputs.
//! * **Monte Carlo estimation of π**: [`mcpi`] module for estimating π using random sampling.

pub(crate) mod _internal;
pub mod chisq;
pub mod mcpi;

pub use chisq::*;
pub use mcpi::*;
