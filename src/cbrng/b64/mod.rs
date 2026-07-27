//! Counter-based 64-bit random number generators.

/// Philox implementations.
pub mod philox;
/// Threefish implementation.
pub mod threefish;

pub use philox::Philox64;
pub use threefish::Threefish256;

crate::impl_default_from_seed64!(Philox64, Threefish256,);
