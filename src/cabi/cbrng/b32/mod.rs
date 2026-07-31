//! C-compatible ABI wrappers for the 32-bit counter-based RNGs.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub(crate) mod philox;
pub(crate) mod squares;
pub(crate) mod threefry;

pub use philox::*;
pub use squares::*;
pub use threefry::*;
