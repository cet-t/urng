//! C-compatible ABI wrappers for the 32-bit stateful PRNGs.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

/// JSF C ABI exports.
pub mod jsf;
/// Mersenne Twister and SFMT C ABI exports.
pub mod mersenne;
/// PCG C ABI exports.
pub mod pcg;
/// SFC C ABI exports.
pub mod sfc;
/// SplitMix C ABI exports.
pub mod splitmix;
/// Xoroshiro C ABI exports.
pub mod xoroshiro;
/// Xorshift and XORWOW C ABI exports.
pub mod xorshift;
/// Xoshiro C ABI exports.
pub mod xoshiro;

pub use jsf::*;
pub use mersenne::*;
pub use pcg::*;
pub use sfc::*;
pub use splitmix::*;
pub use xoroshiro::*;
pub use xorshift::*;
pub use xoshiro::*;
