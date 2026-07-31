//! C-compatible ABI wrappers for the 64-bit counter-based RNGs.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub(crate) mod philox;
pub(crate) mod threefish;

pub use philox::*;
pub use threefish::*;
