//! Consolidated 64-bit random number generators.
//!
//! This module groups the 64-bit RNG implementations and re-exports the main generator types.

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

pub use cet::Cet64;
#[allow(deprecated)]
pub use lcg::Lcg64;
pub use mersenne::{Mt1993764, Sfmt1993764};
pub use philox::Philox64;
pub use sfc::{Sfc64, Sfc64x8};
pub use splitmix::SplitMix64;
pub use threefish::Threefish256;
pub use twisted_gfsr::TwistedGFSR;
pub use xoroshiro::{Xoroshiro128Pp, Xoroshiro128Ss};
pub use xorshift::Xorshift64;
pub use xoshiro::{Xoshiro256Pp, Xoshiro256Ss, Xoshiro256Ssx2};
