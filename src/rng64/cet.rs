use crate::rng::Rng64;
use crate::rng64::SplitMix64;
use crate::wrap;
use std::num::Wrapping;

/// A 64-bit Self-made random number generator.
///
/// This generator uses a 4-cell cellular automaton state and a Weyl counter.
/// It is designed for high performance and quality.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Cet64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Cet64 {
    k: Wrapping<u64>,
    v: [Wrapping<u64>; 4],
    c: Wrapping<u64>,
}

impl Cet64 {
    /// Creates a new `Cet64` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let k = seedgen.nextu();
        Self {
            k: wrap!(k),
            v: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
            c: wrap!(1327),
        }
    }
}

impl Rng64 for Cet64 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        let [mut a, mut b, mut c, mut d] = self.v;

        a += a.0.rotate_left(13).wrapping_mul(self.k.0);
        c += c.0.rotate_left(27) ^ self.c.0;

        b ^= a.0.rotate_left(32);
        d ^= c.0.rotate_left(32);

        self.c += 1327;
        self.v = [a, b, c, d];

        (((a ^ b) ^ (c ^ d)) ^ wrap!(182)).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cet64_works() {
        let mut rng = Cet64::new(1);
        assert_eq!(rng.nextu(), 15169567334506313593);
        assert_eq!(rng.nextf(), 0.7143720878069354);
    }
}
