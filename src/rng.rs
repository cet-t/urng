/// A trait for 32-bit random number generators.
///
/// # Examples
///
/// ```
/// use urng::rng::Rng32;
/// use urng::rng32::Xorshift32;
///
/// let mut rng = Xorshift32::new(1);
/// let val = rng.randi(1, 6);
/// assert!((1..=6).contains(&val));
/// assert!(rng.randf(0.0, 1.0_f32) < 1.0);
/// let items = ["a", "b", "c"];
/// assert!(items.contains(rng.choice(&items)));
/// ```
pub trait Rng32 {
    /// Generates the next random `u32` value in the range [0, 2^32).
    fn nextu(&mut self) -> u32;

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline(always)]
    fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline(always)]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline(always)]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline(always)]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

/// A trait for 64-bit random number generators.
///
/// # Examples
///
/// ```
/// use urng::rng::Rng64;
/// use urng::rng64::Xorshift64;
///
/// let mut rng = Xorshift64::new(1);
/// let val = rng.randi(1, 100);
/// assert!((1..=100).contains(&val));
/// assert!(rng.randf(0.0, 1.0_f64) < 1.0);
/// let items = [10u64, 20, 30];
/// assert!(items.contains(rng.choice(&items)));
/// ```
pub trait Rng64 {
    /// Generates the next random `u64` value in the range [0, 2^64).
    fn nextu(&mut self) -> u64;

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline(always)]
    fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline(always)]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline(always)]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline(always)]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}
