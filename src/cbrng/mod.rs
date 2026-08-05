//! Counter-based random number generators.
//!
//! Unlike the stateful generators in [`crate::prng`],
//! these produce output as a keyed bijection of an explicit counter,
//! so any position in the stream can be evaluated directly.

#![allow(ambiguous_glob_reexports)]

pub(crate) mod b32;
pub(crate) mod b64;

pub use b32::*;
pub use b64::*;
