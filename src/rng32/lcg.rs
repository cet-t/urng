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
/// use urng::rng::Rng32;
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
}

#[allow(deprecated)]
impl Rng32 for Lcg32 {
    /// Generates the next random `u32` value.
    #[inline]
    fn nextu(&mut self) -> u32 {
        // X(n+1) = (a * X(n) + b) % M
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
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
        assert_eq!(rng.nextf(), 1.3969839e-9);
    }
}
