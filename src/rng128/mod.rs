//! 128-bit-state random number generators.

/// Xorshift128 implementation.
pub mod xorshift;

pub use xorshift::Xorshift128;
