//! Stateful pseudo-random number generators.
//!
//! Generators whose output depends on an internal state that is advanced on every
//! draw. Counter-based generators (stateless keyed functions of a counter) live in
//! [`crate::cbrng`].

#![allow(ambiguous_glob_reexports)]

pub mod b32;
pub mod b64;

pub use b32::*;
pub use b64::*;
