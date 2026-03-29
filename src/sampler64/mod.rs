//! Weighted random selection helpers for 64-bit RNGs.

/// Alias method sampler for 64-bit weights.
pub mod alias;
/// Binary-search-tree sampler for 64-bit weights.
pub mod bst;

pub use alias::Alias64;
pub use bst::Bst64;
