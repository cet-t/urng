//! C-compatible ABI wrappers for the 64-bit RNGs.

/// Biski C ABI exports.
pub mod biski;
/// CET C ABI exports.
pub mod cet;
/// LCG C ABI exports.
pub mod lcg;
/// Mersenne Twister and SFMT C ABI exports.
pub mod mersenne;
/// Philox C ABI exports.
pub mod philox;
/// SFC C ABI exports.
pub mod sfc;
/// SplitMix C ABI exports.
pub mod splitmix;
/// Threefish C ABI exports.
pub mod threefish;
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
pub use lcg::*;
pub use mersenne::*;
pub use philox::*;
pub use sfc::*;
pub use splitmix::*;
pub use threefish::*;
pub use twisted_gfsr::*;
pub use xoroshiro::*;
pub use xorshift::*;
pub use xoshiro::*;
