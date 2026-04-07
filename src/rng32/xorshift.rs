use crate::rng32::SplitMix32;
use crate::{rng::Rng32, wrap};
use std::num::Wrapping;

// --- Xorshift32 ---

/// A 32-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xorshift32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorshift32 {
    a: Wrapping<u32>,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance seeded with the given value.
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            a: wrap!(sm.nextu()),
        }
    }
}

impl Rng32 for Xorshift32 {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let x = self.a;
        self.a = x ^ (x << 13);
        self.a ^= self.a >> 17;
        self.a ^= self.a << 5;
        self.a.0
    }
}

// --- Xorshift128 ---

/// A 128-bit Xorshift random number generator.
///
/// Produces 32-bit output from a 128-bit internal state.
/// Period: 2^128 - 1.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xorshift128::new(1);
/// let _ = rng.nextu();
/// assert!(rng.randi(1, 100) >= 1);
/// ```
#[repr(C)]
pub struct Xorshift128 {
    x: [u32; 4],
}

impl Xorshift128 {
    /// Creates a new `Xorshift128` instance.
    ///
    /// Each seed element is OR-ed with 1 to prevent an all-zero state.
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: [sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu()],
        }
    }
}

impl Rng32 for Xorshift128 {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let mut t = self.x[3];
        t ^= t << 11;
        t ^= t >> 8;
        let s = self.x[0];
        (self.x[1], self.x[2], self.x[3]) = (s, self.x[1], self.x[2]);
        self.x[0] = t ^ s ^ (s >> 19);
        self.x[0]
    }
}

/// A XORWOW random number generator.
///
/// This generator combines a Xorshift-based algorithm with a Weyl sequence (linear counter).
/// It has a state of 192 bits (5 x 32-bit state + 32-bit counter).
/// This is the default generator used in NVIDIA cuRAND.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xorwow::new(1);
/// assert_eq!(rng.nextu(), 3932718581);
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [Wrapping<u32>; 5],
    c: Wrapping<u32>,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance seeded with the given value.
    ///
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu()],
            c: wrap!(sm.nextu()),
        }
    }
}

impl Rng32 for Xorwow {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let mut t = self.x[4];

        let s = self.x[0];
        self.x[4] = self.x[3];
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = s;

        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.x[0] = t;
        self.c += wrap!(362437);
        (t + self.c).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift32_works() {
        let mut rng = Xorshift32::new(1);
        assert_eq!(rng.nextu(), 2076024533);
        assert_eq!(rng.nextf(), 0.7677616);
    }

    #[test]
    fn xorwow_works() {
        let mut rng = Xorwow::new(1);
        assert_eq!(rng.nextu(), 3932718581);
        assert_eq!(rng.nextf(), 0.10210251);
    }
}
