//! Counter-based 64-bit random number generators.

pub(crate) mod philox;
pub(crate) mod threefish;

pub use philox::Philox64;
pub use threefish::Threefish256;

crate::impl_default_from_seed64!(Philox64, Threefish256,);
