use wrapn::Wrap;

use crate::rng::Rng;

// --- Lcg64 ---

/// A Linear Congruential Generator (LCG) for 64-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
///
/// # Examples
///
/// ```
/// use urng::rng::Rng;
/// use urng::rng64::Lcg64;
///
/// let mut rng = Lcg64::new(8, 13, 5, 24);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[deprecated(since = "0.2.4", note = "Use Xoshiro256++/** instead.")]
pub struct Lcg64 {
    x: Wrap<u64>,
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
            x: x.into(),
            a: a | 1,
            b,
            m,
            r: 1.0 / (m as f64 + 1.0),
        }
    }
}

#[allow(deprecated)]
impl Rng for Lcg64 {
    type Word = u64;
    #[inline]
    fn nextu(&mut self) -> u64 {
        self.x = (self.x * self.a + self.b) % self.m;
        self.x.value()
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    crate::safe_test!(Lcg64, Lcg64::new(8, 13, 5, 24));
}
