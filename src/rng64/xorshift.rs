use crate::rng::Rng64;
use crate::rng64::SplitMix64;

// --- Xorshift64 ---

/// A 64-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
///
/// # Examples
///
/// ```
/// use urng::rng64::Xorshift64;
///
/// let mut rng = Xorshift64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorshift64 {
    a: u64,
}

impl Xorshift64 {
    /// Creates a new `Xorshift64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self { a: seedgen.nextu() }
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let mut x = self.a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.a = x;
        x
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
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

impl Rng64 for Xorshift64 {
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
    fn xorshift64_works() {
        let mut rng = Xorshift64::new(1);
        assert_eq!(rng.nextu(), 8247328468710148152);
        assert_eq!(rng.nextf(), 0.8223768786697171);
    }
}
