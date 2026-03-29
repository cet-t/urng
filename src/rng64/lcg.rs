use crate::rng::Rng64;
use crate::wrap;
use std::num::Wrapping;

// --- Lcg64 ---

/// A Linear Congruential Generator (LCG) for 64-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
#[repr(C, align(64))]
#[deprecated(since = "0.2.4", note = "Use Xoshiro256++/** instead.")]
pub struct Lcg64 {
    x: Wrapping<u64>,
    a: u64,
    b: u64,
    m: u64,
    r: f64,
}

#[allow(deprecated)]
impl Lcg64 {
    /// Creates a new `Lcg64` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - The initial state (seed).
    /// * `a` - The multiplier.
    /// * `b` - The increment.
    /// * `m` - The modulus.
    pub fn new(x: u64, a: u64, b: u64, m: u64) -> Self {
        Self {
            x: wrap!(x),
            a: a | 1,
            b,
            m,
            r: 1.0 / (m as f64 + 1.0),
        }
    }

    /// Generates the next raw `u64` value via the LCG recurrence.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    /// Generates the next `f64` value in `[0, 1)`.
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * self.r
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
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

#[allow(deprecated)]
impl Rng64 for Lcg64 {
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

    #[allow(deprecated)]
    #[test]
    fn lcg64_works() {
        let mut rng = Lcg64::new(8, 13, 5, 24);
        assert_eq!(rng.nextu(), 13);
        assert_eq!(rng.nextf(), 0.24);
    }
}
