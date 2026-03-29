use crate::rng::Rng64;
use crate::rng64::SplitMix64;
use crate::wrap;
use std::num::Wrapping;

/// A 64-bit Self-made random number generator.
///
/// This generator uses a 4-cell cellular automaton state and a Weyl counter.
/// It is designed for high performance and quality.
#[repr(C)]
pub struct Cet64 {
    k: Wrapping<u64>,
    v: [Wrapping<u64>; 4],
    c: Wrapping<u64>,
}

impl Cet64 {
    /// Creates a new `Cet64` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let k = seedgen.nextu();
        Self {
            k: wrap!(k),
            v: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
            c: wrap!(1327),
        }
    }

    /// Generates the next random `u64` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Cet64;
    ///
    /// let mut rng = Cet64::new(1);
    /// assert_eq!(rng.nextu(), 15169567334506313593);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// let f = rng.randf(0.0, 1.0);
    /// assert!(f >= 0.0 && f < 1.0);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let [mut a, mut b, mut c, mut d] = self.v;

        a += a.0.rotate_left(13).wrapping_mul(self.k.0);
        c += c.0.rotate_left(27) ^ self.c.0;

        b ^= a.0.rotate_left(32);
        d ^= c.0.rotate_left(32);

        self.c += 1327;
        self.v = [a, b, c, d];

        (((a ^ b) ^ (c ^ d)) ^ wrap!(182)).0
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
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Cet64 {
    #[inline]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cet64_works() {
        let mut rng = Cet64::new(1);
        assert_eq!(rng.nextu(), 15169567334506313593);
        assert_eq!(rng.nextf(), 0.7143720878069354);
    }
}
