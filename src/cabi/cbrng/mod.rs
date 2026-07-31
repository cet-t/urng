//! C-compatible ABI wrappers for the counter-based RNGs.

#![allow(ambiguous_glob_reexports)]

pub(crate) mod b32;
pub(crate) mod b64;

pub use b32::*;
pub use b64::*;
