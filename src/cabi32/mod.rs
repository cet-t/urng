//! C-compatible ABI wrappers for the 32-bit RNGs.

/// LCG C ABI exports.
pub mod lcg;
/// Mersenne Twister and SFMT C ABI exports.
pub mod mersenne;
/// PCG C ABI exports.
pub mod pcg;
/// Philox C ABI exports.
pub mod philox;
/// SplitMix C ABI exports.
pub mod splitmix;
/// Squares C ABI exports.
pub mod squares;
/// Threefry C ABI exports.
pub mod threefry;
/// Xorshift and XORWOW C ABI exports.
pub mod xorshift;
/// Xoshiro C ABI exports.
pub mod xoshiro;

pub use lcg::*;
pub use mersenne::*;
pub use pcg::*;
pub use philox::*;
pub use splitmix::*;
pub use squares::*;
pub use threefry::*;
pub use xorshift::*;
pub use xoshiro::*;
