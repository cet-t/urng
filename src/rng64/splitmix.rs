use crate::rng::Rng64;
use crate::wrap;
use std::num::Wrapping;

/// A SplitMix64 random number generator.
///
/// This is a fast generator with 64-bit state, often used for initializing
/// other generators from a single seed.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = SplitMix64::new(1);
/// assert_eq!(rng.nextu(), 10451216379200822465);
/// ```
#[repr(align(64))]
pub struct SplitMix64 {
    pub(crate) s: Wrapping<u64>,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` instance.
    pub fn new(seed: u64) -> Self {
        Self { s: wrap!(seed | 1) }
    }

    /// Computes the SplitMix64 output for a given raw state word (pure, stateless).
    #[inline]
    pub(crate) fn compute(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl Rng64 for SplitMix64 {
    #[inline]
    fn nextu(&mut self) -> u64 {
        self.s += 0x9E3779B97F4A7C15;
        Self::compute(self.s.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(SplitMix64);
}
