//! Weighted random selection helpers for 32-bit RNGs.

/// Alias method sampler for 32-bit weights.
pub mod alias;
/// Binary-search-tree sampler for 32-bit weights.
pub mod bst;

pub use alias::Alias32;
pub use bst::Bst32;
