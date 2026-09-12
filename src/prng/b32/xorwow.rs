use wrapn::{wrap, wu32};

use crate::{Rng, SplitMix32};

/// A XORWOW random number generator.
///
/// This generator combines a Xorshift-based algorithm with a Weyl sequence (linear counter).
/// It has a state of 192 bits (5 x 32-bit state + 32-bit counter).
/// This is the default generator used in NVIDIA cuRAND.
///
/// # Example
/// ```
/// use urng::{Rng, Xorwow};
///
/// let mut rng = Xorwow::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Xorwow {
    x: [wu32; 5],
    c: wu32,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance with the given seed.
    pub const fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![
                sm.nextu_const(),
                sm.nextu_const(),
                sm.nextu_const(),
                sm.nextu_const(),
                sm.nextu_const()
            ],
            c: wrap!(sm.nextu_const()),
        }
    }
}

impl Rng for Xorwow {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
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
        self.c += 362437;
        *(t + self.c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! { Xorwow }
}
