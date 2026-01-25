/// A trait for 32-bit random number generators.
pub trait Rng32 {
    /// Generates a random `i32` value in the range [min, max].
    fn randi(&mut self, min: i32, max: i32) -> i32;
    /// Generates a random `f32` value in the range [min, max).
    fn randf(&mut self, min: f32, max: f32) -> f32;
    /// Returns a random element from a slice.
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T;
}

/// A trait for 64-bit random number generators.
pub trait Rng64 {
    /// Generates a random `i64` value in the range [min, max].
    fn randi(&mut self, min: i64, max: i64) -> i64;
    /// Generates a random `f64` value in the range [min, max).
    fn randf(&mut self, min: f64, max: f64) -> f64;
    /// Returns a random element from a slice.
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T;
}
