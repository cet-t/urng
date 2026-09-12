use wrapn::{wrap, wu64};

use crate::rng::Rng;

/// SplitMix64 64-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::{Rng, SplitMix64};
///
/// let mut rng = SplitMix64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64 {
    pub(crate) s: wu64,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` instance with the given seed.
    pub const fn new(seed: u64) -> Self {
        Self { s: wrap!(seed | 1) }
    }

    /// Computes the SplitMix64 output for a given raw state word (pure, stateless).
    #[inline]
    pub(crate) const fn compute(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline(always)]
    pub(crate) const fn nextu_const(&mut self) -> u64 {
        self.s.0.0 = self.s.0.0.wrapping_add(0x9E3779B97F4A7C15);
        Self::compute(self.s.0.0)
    }
}

impl Rng for SplitMix64 {
    type Word = u64;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        self.s += 0x9E3779B97F4A7C15;
        Self::compute(self.s.0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! { SplitMix64 }
}
