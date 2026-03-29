use crate::rng32::SplitMix32;
use crate::{rng::Rng32, wrap};
use std::num::Wrapping;

// --- Xorshift32 ---

/// A 32-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
///
/// # Examples
///
/// ```
/// use urng::rng32::Xorshift32;
///
/// let mut rng = Xorshift32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorshift32 {
    a: Wrapping<u32>,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
    /// assert_eq!(rng.nextu(), 2076024533);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            a: wrap!(sm.nextu()),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
    /// assert_eq!(rng.nextu(), 2076024533);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let x = self.a;
        self.a = x ^ (x << 13);
        self.a ^= self.a >> 17;
        self.a ^= self.a << 5;
        self.a.0
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
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

impl Rng32 for Xorshift32 {
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

/// A XORWOW random number generator.
///
/// This generator combines a Xorshift-based algorithm with a Weyl sequence (linear counter).
/// It has a state of 192 bits (5 x 32-bit state + 32-bit counter).
/// This is the default generator used in NVIDIA cuRAND.
///
/// # Examples
///
/// ```
/// use urng::rng32::Xorwow;
///
/// let mut rng = Xorwow::new(1);
/// assert_eq!(rng.nextu(), 3932718581);
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [Wrapping<u32>; 5],
    c: Wrapping<u32>,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
    /// assert_eq!(rng.nextu(), 3932718581);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu()],
            c: wrap!(sm.nextu()),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
    /// assert_eq!(rng.nextu(), 3932718581);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let mut t = self.x[4];

        let s = self.x[0];
        self.x[4] = self.x[3];
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = s;

        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.x[0] = t;
        self.c += wrap!(362437);
        (t + self.c).0
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
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
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
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
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
    /// let items = ["red", "green", "blue"];
    /// assert!(items.contains(rng.choice(&items)));
    /// ```
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Xorwow {
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }

    fn randf(&mut self, min: f32, max: f32) -> f32 {
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
    fn xorshift32_works() {
        let mut rng = Xorshift32::new(1);
        assert_eq!(rng.nextu(), 2076024533);
        assert_eq!(rng.nextf(), 0.7677616);
    }

    #[test]
    fn xorwow_works() {
        let mut rng = Xorwow::new(1);
        assert_eq!(rng.nextu(), 3932718581);
        assert_eq!(rng.nextf(), 0.10210251);
    }
}
