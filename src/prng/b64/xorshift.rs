use wrapn::{wrap, wu64};

use crate::prng::b64::SplitMix64;
use crate::rng::Rng;

// --- Xorshift64 ---

/// Xorshift 64-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::{Rng, Xorshift64};
///
/// let mut rng = Xorshift64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Xorshift64 {
    a: wu64,
}

impl Xorshift64 {
    /// Creates a new `Xorshift64` instance with the given seed.
    pub const fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            a: wrap!(seedgen.nextu_const()),
        }
    }
}

impl Rng for Xorshift64 {
    type Word = u64;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        let mut x = self.a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.a = x;
        *x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! { Xorshift64 }
}
