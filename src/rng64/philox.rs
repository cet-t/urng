use crate::{rng::Rng64, rng64::SplitMix64};

// --- Philox64 ---

/// A Philox 2x64 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
///
/// # Examples
///
/// ```
/// use urng::rng64::Philox64;
///
/// let mut rng = Philox64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Philox64 {
    pub(crate) c: [u64; 2],
    pub(crate) k: [u64; 2],
}

impl Philox64 {
    const fn m0() -> u128 {
        0xD2B74407B1CE6E93
    }

    /// Creates a new `Philox64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            c: [1, 0],
            k: [seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline]
    pub(crate) fn compute(c: [u64; 2], k: [u64; 2]) -> [u64; 2] {
        let mut v0 = c[0];
        let mut v1 = c[1];
        let mut key = k[0];

        let w0: u64 = 0x9E3779B97F4A7C15;

        for _ in 0..10 {
            let prod = (v0 as u128).wrapping_mul(Self::m0());
            let hi = (prod >> 64) as u64;
            let lo = prod as u64;
            let next_v0 = hi ^ v1 ^ key;
            let next_v1 = lo;

            v0 = next_v0;
            v1 = next_v1;
            key = key.wrapping_add(w0);
        }

        [v0, v1]
    }

    /// Generates the next block of random numbers.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 2] {
        let out = Self::compute(self.c, self.k);
        self.c[0] = self.c[0].wrapping_add(1);
        if self.c[0] == 0 {
            self.c[1] = self.c[1].wrapping_add(1);
        }
        out
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu()[0] as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu()[0];
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu()[0] as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1) as usize;
        &choices[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Philox64);
}
