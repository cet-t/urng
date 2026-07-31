//! Consolidated 64-bit pseudo-random number generators.
//!
//! This module groups the stateful (non counter-based) 64-bit RNG implementations
//! and re-exports the main generator types. Counter-based generators live in
//! [`crate::cbrng::b64`].

pub(crate) mod biski;
pub(crate) mod cet;
pub(crate) mod mersenne;
pub(crate) mod sfc;
pub(crate) mod splitmix;
pub(crate) mod twisted_gfsr;
pub(crate) mod wyhash;
pub(crate) mod xoroshiro;
pub(crate) mod xorshift;
pub(crate) mod xoshiro;

pub use biski::Biski64;
#[cfg(feature = "simd")]
pub use biski::Biski64x8;
pub use cet::{Cet64, Cet256};
#[cfg(feature = "simd")]
pub use cet::{Cet64x8, Cet256x2};
pub use mersenne::{Mt1993764, Sfmt1993764};
pub use sfc::Sfc64;
#[cfg(feature = "simd")]
pub use sfc::Sfc64x8;
pub use splitmix::SplitMix64;
pub use twisted_gfsr::TwistedGFSR;
pub use wyhash::WyHash64;
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
    WyHash64
);
