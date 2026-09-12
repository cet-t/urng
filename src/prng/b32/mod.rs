//! Consolidated 32-bit pseudo-random number generators.
//!
//! This module groups the stateful (non counter-based) 32-bit RNG implementations
//! and re-exports the main generator types. Counter-based generators live in
//! [`crate::cbrng::b32`].

pub(crate) mod jsf;
pub(crate) mod mersenne;
pub(crate) mod pcg;
pub(crate) mod sfc;
pub(crate) mod splitmix;
pub(crate) mod xoroshiro;
pub(crate) mod xorshift;
pub(crate) mod xorwow;
pub(crate) mod xoshiro;

pub use jsf::Jsf32;
#[cfg(feature = "simd")]
pub use jsf::{Jsf32x8, Jsf32x16};
pub use mersenne::{
    Mt19937, Sfmt607, Sfmt1279, Sfmt2281, Sfmt4253, Sfmt11213, Sfmt19937, Sfmt44497, Sfmt86243,
    Sfmt132049, Sfmt216091,
};
pub use pcg::Pcg32;
#[cfg(feature = "simd")]
pub use pcg::Pcg32x8;
pub use sfc::Sfc32;
#[cfg(feature = "simd")]
pub use sfc::{Sfc32x4, Sfc32x8, Sfc32x16};
pub use splitmix::SplitMix32;
#[cfg(feature = "simd")]
pub use splitmix::SplitMix32x16;
pub use xoroshiro::Xoroshiro64Ss;
#[cfg(feature = "simd")]
pub use xoroshiro::Xoroshiro64Ssx16;
pub use xorshift::{Xorshift32, Xorshift128};
pub use xorwow::Xorwow;
pub use xoshiro::{Xoshiro128Pp, Xoshiro128Ss};
#[cfg(feature = "simd")]
pub use xoshiro::{Xoshiro128Ppx16, Xoshiro128Ssx16};

#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use pcg::{PCG32_MULT, PCG32X8_LANE, PCG32X8_PAR_CHUNK, PCG32X8_PAR_CHUNK_BLOCKS};
#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use splitmix::{SPLITMIX32_GAMMA, SPLITMIX32X16, SPLITMIX32X16_PAR_CHUNK};

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
);
