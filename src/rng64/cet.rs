use std::arch::x86_64::__m512i;

use crate::rng::Rng64;
use crate::rng64::SplitMix64;

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
    s: u64,
}

const SP1: u64 = 0xFFFFFFFFFFFFFF43;
const SP2: u64 = 0xFFFFFFFFFFFFFF1B;
const P1: u64 = 0x94D049BB133111EB;

impl Cet64 {
    /// Creates a new `Cet64` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self { s: seedgen.nextu() }
    }
}

impl Rng64 for Cet64 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        self.s = self.s.wrapping_add(SP1);

        let mut x = self.s;
        x ^= x >> 30;
        x = x.wrapping_mul(SP2);
        x ^= x >> 27;
        x = x.wrapping_mul(P1);
        x ^= x >> 31;

        x
    }
}

pub struct Cet256 {
    s: [u64; 4],
}

impl Cet256 {
    /// Creates a new `Cet256` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: [
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
            ],
        }
    }
}

impl Rng64 for Cet256 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        self.s[0] = self.s[0].wrapping_add(SP1);
        let c0 = (self.s[0] < SP1) as u64;
        self.s[1] = self.s[1].wrapping_add(c0);
        let c1 = (self.s[1] < c0) as u64;
        self.s[2] = self.s[2].wrapping_add(c1);
        let c2 = (self.s[2] < c1) as u64;
        self.s[3] = self.s[3].wrapping_add(c2);

        let mut x = self.s[0] ^ self.s[3];
        x = x.wrapping_add(self.s[1].rotate_left(17));

        x ^= x >> 30;
        x = x.wrapping_mul(SP2);
        x ^= x >> 27;
        x = x.wrapping_mul(P1);
        x ^= x >> 31;

        x
    }
}

pub struct Cet64x8 {
    s: __m512i,
}

pub struct Cet256x2 {
    s: __m512i,
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Cet64);
    crate::safe_test!(Cet256);
}
