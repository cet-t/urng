//! 32-bit cipher-based random number generators (ChaCha, Salsa20).

pub(crate) mod chacha;
pub(crate) mod salsa;

pub use chacha::{ChaCha8, ChaCha20};
pub use salsa::{Salsa8, Salsa20};
