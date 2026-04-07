use crate::rng::Rng64;
use crate::wrap;
use std::num::Wrapping;

// --- Lcg64 ---

/// A Linear Congruential Generator (LCG) for 64-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
///
/// # Examples
///
/// ```
/// use urng::rng::Rng64;
/// use urng::rng64::Lcg64;
///
/// let mut rng = Lcg64::new(8, 13, 5, 24);
/// let _ = rng.nextu();
/// ```
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
    pub fn new(x: u64, a: u64, b: u64, m: u64) -> Self {
        Self {
            x: wrap!(x),
            a: a | 1,
            b,
            m,
            r: 1.0 / (m as f64 + 1.0),
        }
    }
}

#[allow(deprecated)]
impl Rng64 for Lcg64 {
    #[inline]
    fn nextu(&mut self) -> u64 {
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
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
        assert_eq!(rng.nextf(), 3.2526065174565133e-19);
    }
}
