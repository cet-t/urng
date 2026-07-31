//! C-compatible ABI wrappers for the RNGs.

#![allow(ambiguous_glob_reexports)]

pub mod cbrng;
pub mod prng;

pub use cbrng::*;
pub use prng::*;
