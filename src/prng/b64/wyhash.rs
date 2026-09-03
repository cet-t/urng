use wrapn::{wrap, wu64};

use crate::{Rng, SplitMix64};

/// Wyhash 64-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::{Rng, WyHash64};
///
/// let mut rng = WyHash64::new(0);
/// let _ = rng.nextu();
/// ```
pub struct WyHash64 {
    s: wu64,
}

impl WyHash64 {
    /// Creates a new `WyHash64` instance with the given seed.
    pub const fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: wrap!(seedgen.nextu_const()),
        }
    }
}

impl Rng for WyHash64 {
    type Word = u64;

    #[inline(always)]
    fn nextu(&mut self) -> Self::Word {
        self.s += 0x60bee2bee120fc15;
        let mut tmp = self.s.cast::<u128>() * 0xa3b195354a39b70d;
        let m1: wu64 = ((tmp >> 64) ^ tmp).cast();
        tmp = m1.cast::<u128>() * 0x1b03738712fad5c9;
        let m2 = (tmp >> 64) ^ tmp;
        m2.cast::<u64>().value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(WyHash64);
}
