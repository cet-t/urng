//! Cipher-based random number generators.
//!
//! Generators derived from stream ciphers (ChaCha, Salsa20):
//! a keystream generated from a key/counter state, consumed as random output.

#![allow(ambiguous_glob_reexports)]

pub(crate) mod b32;

pub use b32::*;
