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
/// use urng::rng64::SplitMix64;
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
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::SplitMix64;
    ///
    /// let mut rng = SplitMix64::new(1);
    /// assert_eq!(rng.nextu(), 10451216379200822465);
    /// assert_eq!(rng.nextf(), 0.7457817572627012);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// ```
    pub fn new(seed: u64) -> Self {
        Self { s: wrap!(seed | 1) }
    }

    /// Computes the SplitMix64 output for a given raw state word (pure, stateless).
    #[inline]
    pub fn compute(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        self.s += 0x9E3779B97F4A7C15;
        Self::compute(self.s.0)
    }

    /// Generates the next `f64` value in `[0, 1)`.
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for SplitMix64 {
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }

    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }

    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_works() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(rng.nextu(), 10451216379200822465);
        assert_eq!(rng.nextf(), 0.7457817572627012);
    }
}
