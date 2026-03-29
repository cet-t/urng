use crate::{rng::Rng32, wrap};
use std::num::Wrapping;

// --- Lcg32 ---

/// A Linear Congruential Generator (LCG) for 32-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
///
/// # Examples
///
/// ```
/// use urng::rng32::Lcg32;
///
/// let mut rng = Lcg32::new(8, 13, 5, 24);
/// let _ = rng.nextu();
/// ```
#[deprecated(since = "0.2.4", note = "Use Xoshiro256++/** instead.")]
#[repr(C)]
pub struct Lcg32 {
    x: Wrapping<u32>,
    a: u32,
    b: u32,
    m: u32,
    r: f32,
}

#[allow(deprecated)]
impl Lcg32 {
    /// Creates a new `Lcg32` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - The initial state (seed).
    /// * `a` - The multiplier.
    /// * `b` - The increment.
    /// * `m` - The modulus.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24);
    /// assert_eq!(rng.nextu(), 13);
    /// ```
    pub fn new(x: u32, a: u32, b: u32, m: u32) -> Self {
        // M>a, M>b, A>0, B>=0
        Self {
            x: wrap!(x),
            a: a | 1,
            b,
            m,
            r: 1.0 / (m as f32 + 1.0),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24);
    /// assert_eq!(rng.nextu(), 13);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        // X(n+1) = (a * X(n) + b) % M
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * self.r
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24);
    /// let val: f32 = rng.randf(0.0, 1.0);
    /// assert!(val >= 0.0 && val < 1.0);
    /// ```
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

#[allow(deprecated)]
impl Rng32 for Lcg32 {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[allow(deprecated)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg32_works() {
        let mut rng = Lcg32::new(8, 13, 5, 24);
        assert_eq!(rng.nextu(), 13);
        assert_eq!(rng.nextf(), 0.24);
    }
}
