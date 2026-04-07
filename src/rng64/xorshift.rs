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
/// use urng::rng::Rng64;
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
}

impl Rng64 for Xorshift64 {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let mut x = self.a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.a = x;
        x
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
