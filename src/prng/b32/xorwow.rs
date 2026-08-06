use wrapn::{wrap, wu32};

use crate::{Rng, SplitMix32};

/// A XORWOW random number generator.
///
/// This generator combines a Xorshift-based algorithm with a Weyl sequence (linear counter).
/// It has a state of 192 bits (5 x 32-bit state + 32-bit counter).
/// This is the default generator used in NVIDIA cuRAND.
///
/// # Examples
///
/// ```
/// use urng::{Rng, Xorwow};
///
/// let mut rng = Xorwow::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [wu32; 5],
    c: wu32,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance seeded with the given value.
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu()],
            c: wrap!(sm.nextu()),
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
        (t + self.c).value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xorwow);
}
