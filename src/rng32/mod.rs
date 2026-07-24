#![allow(deprecated)]
//! Consolidated 32-bit random number generators.
//!
//! This module groups the 32-bit RNG implementations and re-exports the main generator types.

/// Jenkins Small Fast implementations.
pub mod jsf;
/// Linear Congruential Generator implementations.
pub mod lcg;
/// Mersenne Twister and SFMT implementations.
pub mod mersenne;
/// PCG implementations, including SIMD variants.
pub mod pcg;
/// Philox implementations, including SIMD variants.
pub mod philox;
/// SFC implementations, including SIMD variants.
pub mod sfc;
/// SplitMix implementations, including SIMD variants.
pub mod splitmix;
/// Squares implementations, including SIMD variants.
pub mod squares;
/// Threefry implementations.
pub mod threefry;
/// Xoroshiro implementations.
pub mod xoroshiro;
/// Xorshift and XORWOW implementations.
pub mod xorshift;
/// Xoshiro implementations.
pub mod xoshiro;

pub use jsf::Jsf32;
#[cfg(feature = "simd")]
pub use jsf::{Jsf32x8, Jsf32x16};
pub use lcg::Lcg32;
pub use mersenne::{
    Mt19937, Sfmt607, Sfmt1279, Sfmt2281, Sfmt4253, Sfmt11213, Sfmt19937, Sfmt44497, Sfmt86243,
    Sfmt132049, Sfmt216091,
};
pub use pcg::Pcg32;
#[cfg(feature = "simd")]
pub use pcg::{Pcg32Simd, Pcg32x8};
pub use philox::Philox32x4;
#[cfg(feature = "simd")]
pub use philox::{Philox32, Philox32x4x4};
pub use sfc::Sfc32;
#[cfg(feature = "simd")]
pub use sfc::{Sfc32x4, Sfc32x8, Sfc32x16};
pub use splitmix::SplitMix32;
#[cfg(feature = "simd")]
pub use splitmix::{SplitMix32Simd, SplitMix32x16};
pub use squares::Squares32;
#[cfg(feature = "simd")]
pub use squares::{Squares32Simd, Squares32x8};
pub use threefry::{Threefry32x2, Threefry32x4};
pub use xoroshiro::Xoroshiro64Ss;
#[cfg(feature = "simd")]
pub use xoroshiro::Xoroshiro64Ssx16;
pub use xorshift::{Xorshift32, Xorshift128, Xorwow};
pub use xoshiro::{Xoshiro128Pp, Xoshiro128Ss};
#[cfg(feature = "simd")]
pub use xoshiro::{Xoshiro128Ppx16, Xoshiro128Ssx16};

#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use pcg::{PCG32_MULT, PCG32X8_LANE, PCG32X8_PAR_CHUNK, PCG32X8_PAR_CHUNK_BLOCKS};
#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use philox::{
    PHILOX32x4x4_CHUNK_RATIO, PHILOX32x4x4_PAR_CHUNK, PHILOX32x4x4_SHIFT, PHILOX32x16,
    PHILOX32x16_SHIFT,
};
#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use splitmix::{SPLITMIX32_GAMMA, SPLITMIX32x16, SPLITMIX32x16_PAR_CHUNK};
#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use squares::SQUARES32x8;

crate::impl_default_from_seed32!(
    Jsf32,
    Mt19937,
    Sfc32,
    SplitMix32,
    Xoroshiro64Ss,
    Xorshift32,
    Xorshift128,
    Xorwow,
    Xoshiro128Pp,
    Xoshiro128Ss,
    Squares32,
    Pcg32,
    Sfmt19937,
    Sfmt607,
    Sfmt1279,
    Sfmt2281,
    Sfmt4253,
    Sfmt11213,
    Sfmt44497,
    Sfmt86243,
    Sfmt132049,
    Sfmt216091,
    Philox32x4,
    Threefry32x2,
    Threefry32x4,
);
