//! Consolidated 64-bit random number generators.
//!
//! This module groups the 64-bit RNG implementations and re-exports the main generator types.

/// Biski generator implementation.
pub mod biski;
/// Cellular automaton and custom experimental generators.
pub mod cet;
/// Linear Congruential Generator implementations.
pub mod lcg;
/// Mersenne Twister and SFMT implementations.
pub mod mersenne;
/// Philox implementations.
pub mod philox;
/// Small Fast Chaotic generator implementations.
pub mod sfc;
/// SplitMix implementations.
pub mod splitmix;
/// Threefish implementation.
pub mod threefish;
/// Twisted Generalized Feedback Shift Register implementation.
pub mod twisted_gfsr;
/// Xoroshiro implementations.
pub mod xoroshiro;
/// Xorshift implementations.
pub mod xorshift;
/// Xoshiro implementations.
pub mod xoshiro;

pub use biski::Biski64;
#[cfg(feature = "simd")]
pub use biski::Biski64x8;
pub use cet::{Cet64, Cet256};
#[cfg(feature = "simd")]
pub use cet::{Cet64x8, Cet256x2};
#[allow(deprecated)]
pub use lcg::Lcg64;
pub use mersenne::{Mt1993764, Sfmt1993764};
pub use philox::Philox64;
pub use sfc::Sfc64;
#[cfg(feature = "simd")]
pub use sfc::Sfc64x8;
pub use splitmix::SplitMix64;
pub use threefish::Threefish256;
pub use twisted_gfsr::TwistedGFSR;
pub use xoroshiro::{Xoroshiro128Pp, Xoroshiro128Ss};
pub use xorshift::Xorshift64;
#[cfg(feature = "simd")]
pub use xoshiro::Xoshiro256Ssx2;
pub use xoshiro::{Xoshiro256Pp, Xoshiro256Ss};

crate::impl_default_from_seed64!(
    Biski64,
    Cet64,
    Cet256,
    Mt1993764,
    Sfmt1993764,
    Sfc64,
    SplitMix64,
    TwistedGFSR,
    Xoroshiro128Pp,
    Xoroshiro128Ss,
    Xorshift64,
    Xoshiro256Pp,
    Xoshiro256Ss,
);
