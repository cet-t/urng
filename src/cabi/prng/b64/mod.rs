//! C-compatible ABI wrappers for the 64-bit stateful PRNGs.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

/// Biski C ABI exports.
pub mod biski;
/// CET C ABI exports.
pub mod cet;
/// Mersenne Twister and SFMT C ABI exports.
pub mod mersenne;
/// SFC C ABI exports.
pub mod sfc;
/// SplitMix C ABI exports.
pub mod splitmix;
/// Twisted GFSR C ABI exports.
pub mod twisted_gfsr;
/// Xoroshiro C ABI exports.
pub mod xoroshiro;
/// Xorshift C ABI exports.
pub mod xorshift;
/// Xoshiro C ABI exports.
pub mod xoshiro;

pub use biski::*;
pub use cet::*;
pub use mersenne::*;
pub use sfc::*;
pub use splitmix::*;
pub use twisted_gfsr::*;
pub use xoroshiro::*;
pub use xorshift::*;
pub use xoshiro::*;
