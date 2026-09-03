use wrapn::{wrap, wu32};

use crate::prng::b32::SplitMix32;
use crate::rng::Rng;

// --- Xorshift32 ---

/// A 32-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
///
/// # Example
/// ```
/// use urng::{Rng, Xorshift32};
///
/// let mut rng = Xorshift32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorshift32 {
    a: wu32,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance with the given seed.
    pub const fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            a: wrap!(sm.nextu_const()),
        }
    }
}

impl Rng for Xorshift32 {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        let x = self.a;
        self.a = x ^ (x << 13);
        self.a ^= self.a >> 17;
        self.a ^= self.a << 5;
        self.a.value()
    }
}

// --- Xorshift128 ---

/// A 128-bit Xorshift random number generator.
///
/// Produces 32-bit output from a 128-bit internal state.
/// Period: 2^128 - 1.
///
/// # Example
/// ```
/// use urng::{Rng, Xorshift128};
///
/// let mut rng = Xorshift128::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorshift128 {
    x: [wu32; 4],
}

impl Xorshift128 {
    /// Creates a new `Xorshift128` instance.
    ///
    /// Each seed element is OR-ed with 1 to prevent an all-zero state.
    pub const fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![
                sm.nextu_const(),
                sm.nextu_const(),
                sm.nextu_const(),
                sm.nextu_const()
            ],
        }
    }
}

impl Rng for Xorshift128 {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        let mut t = self.x[3];
        t ^= t << 11;
        t ^= t >> 8;
        let s = self.x[0];
        (self.x[1], self.x[2], self.x[3]) = (s, self.x[1], self.x[2]);
        self.x[0] = t ^ s ^ (s >> 19);
        self.x[0].value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xorshift32);
    crate::safe_test!(Xorshift128);
}
