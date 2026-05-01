//! C-compatible ABI wrappers for the 32-bit RNGs.

/// JSF C ABI exports.
pub mod jsf;
/// LCG C ABI exports.
pub mod lcg;
/// Mersenne Twister and SFMT C ABI exports.
pub mod mersenne;
/// PCG C ABI exports.
pub mod pcg;
/// Philox C ABI exports.
pub mod philox;
/// SFC C ABI exports.
pub mod sfc;
/// SplitMix C ABI exports.
pub mod splitmix;
/// Squares C ABI exports.
pub mod squares;
/// Threefry C ABI exports.
pub mod threefry;
/// Xoroshiro C ABI exports.
pub mod xoroshiro;
/// Xorshift and XORWOW C ABI exports.
pub mod xorshift;
/// Xoshiro C ABI exports.
pub mod xoshiro;

pub use jsf::*;
pub use lcg::*;
pub use mersenne::*;
pub use pcg::*;
pub use philox::*;
pub use sfc::*;
pub use splitmix::*;
pub use squares::*;
pub use threefry::*;
pub use xoroshiro::*;
pub use xorshift::*;
pub use xoshiro::*;
