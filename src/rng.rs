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
    /// Generates a random `i32` value in the range [min, max].
    fn randi(&mut self, min: i32, max: i32) -> i32;
    /// Generates a random `f32` value in the range [min, max).
    fn randf(&mut self, min: f32, max: f32) -> f32;
    /// Returns a random element from a slice.
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T;
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
    /// Generates a random `i64` value in the range [min, max].
    fn randi(&mut self, min: i64, max: i64) -> i64;
    /// Generates a random `f64` value in the range [min, max).
    fn randf(&mut self, min: f64, max: f64) -> f64;
    /// Returns a random element from a slice.
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T;
}
