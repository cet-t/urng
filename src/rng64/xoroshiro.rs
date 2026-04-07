use crate::{rng::Rng64, rng64::SplitMix64, wrap};
use std::num::Wrapping;

/// A xoshiro128++ random number generator.
///
/// This is a fast 128-bit-state generator with good statistical quality.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xoroshiro128Pp::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoroshiro128Pp {
    s: [Wrapping<u64>; 2],
}

impl Xoroshiro128Pp {
    /// Creates a new `Xoroshiro128Pp` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Self {
            s: [Wrapping(seedgen.nextu()), Wrapping(seedgen.nextu())],
        }
    }

    /// Applies the jump function for advancing by $2^{64}$ steps.
    #[inline]
    pub fn jump(&mut self) {
        const JUMP: [u64; 2] = [0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05];

        macro_rules! jump {
            ($i:expr, $s0:ident, $s1:ident) => {
                // TODO: unrolling
                for b in 0..64 {
                    if (JUMP[$i] & (1 << b)) != 0 {
                        $s0 ^= self.s[0].0;
                        $s1 ^= self.s[1].0;
                    }
                    self.nextu();
                }
            };
        }

        let mut s0 = 0;
        let mut s1 = 0;
        jump!(0, s0, s1);
        jump!(1, s0, s1);

        self.s = wrap![s0, s1];
    }

    /// Applies the long-jump function for advancing by $2^{96}$ steps.
    #[inline]
    pub fn long_jump(&mut self) {
        const LONG_JUMP: [u64; 2] = [0x360fd5f2cf8d5d99, 0x9c6e6877736c46e3];

        macro_rules! long_jump {
            ($i:expr, $s0:ident, $s1:ident) => {
                for i in 0..LONG_JUMP.len() {
                    for b in 0..64 {
                        if (LONG_JUMP[i] & (1 << b)) != 0 {
                            $s0 ^= self.s[0].0;
                            $s1 ^= self.s[1].0;
                        }
                        self.nextu();
                    }

                    self.s = wrap![$s0, $s1];
                }
            };
        }

        let mut s0 = 0;
        let mut s1 = 0;
        long_jump!(0, s0, s1);
        long_jump!(1, s0, s1);

        self.s = wrap![s0, s1];
    }
}

impl Rng64 for Xoroshiro128Pp {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let s0 = self.s[0];
        let s1 = self.s[1];
        let result = s0 + s1;

        let s0 = s0.0;
        let s1 = s1.0;

        self.s = wrap![
            s1 ^ s0,
            s0.rotate_left(24) ^ self.s[1].0 ^ (self.s[1].0 << 16)
        ];
        self.s[1] = wrap!(self.s[1].0.rotate_left(37));

        result.0
    }
}

/// A xoshiro128** random number generator.
///
/// This is a fast 128-bit-state generator with good statistical properties.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xoroshiro128Ss::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoroshiro128Ss {
    s: [Wrapping<u64>; 2],
}

impl Xoroshiro128Ss {
    /// Creates a new `Xoroshiro128Ss` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Self {
            s: [Wrapping(seedgen.nextu()), Wrapping(seedgen.nextu())],
        }
    }

    /// Applies the jump function for advancing by $2^{64}$ steps.
    #[inline]
    pub fn jump(&mut self) {
        const JUMP: [u64; 2] = [0xdf900294d8f554a5, 0x170865df4b3201fc];

        macro_rules! jump {
            ($i:expr, $s0:ident, $s1:ident) => {
                // TODO: unrolling
                for b in 0..64 {
                    if (JUMP[$i] & (1 << b)) != 0 {
                        $s0 ^= self.s[0].0;
                        $s1 ^= self.s[1].0;
                    }
                    self.nextu();
                }
            };
        }

        let mut s0 = 0;
        let mut s1 = 0;
        jump!(0, s0, s1);
        jump!(1, s0, s1);

        self.s = wrap![s0, s1];
    }

    /// Applies the long-jump function for advancing by $2^{96}$ steps.
    #[inline]
    pub fn long_jump(&mut self) {
        const LONG_JUMP: [u64; 2] = [0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1];

        macro_rules! long_jump {
            ($i:expr, $s0:ident, $s1:ident) => {
                for i in 0..LONG_JUMP.len() {
                    for b in 0..64 {
                        if (LONG_JUMP[i] & (1 << b)) != 0 {
                            $s0 ^= self.s[0].0;
                            $s1 ^= self.s[1].0;
                        }
                        self.nextu();
                    }

                    self.s = wrap![$s0, $s1];
                }
            };
        }

        let mut s0 = 0;
        let mut s1 = 0;
        long_jump!(0, s0, s1);
        long_jump!(1, s0, s1);

        self.s = wrap![s0, s1];
    }
}

impl Rng64 for Xoroshiro128Ss {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let s0 = self.s[0];
        let s1 = self.s[1];
        let result = wrap!((s0 * wrap!(5)).0.rotate_left(7)) * wrap!(9);

        let s1 = s1 ^ s0;
        self.s[0] = wrap!(s0.0.rotate_left(24) ^ s1.0 ^ (s1.0 << 16));
        self.s[1] = wrap!(s1.0.rotate_left(37));

        result.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xoroshiro128pp() {
        let mut rng = Xoroshiro128Pp::new(12345);
        assert_eq!(rng.nextu(), 6233086606872742541);
        assert_eq!(rng.nextf(), 0.07074813251086551);
    }

    #[test]
    fn test_xoroshiro128ss() {
        let mut rng = Xoroshiro128Ss::new(12345);
        assert_eq!(rng.nextu(), 9940793396233540349);
        assert_eq!(rng.nextf(), 0.47619897611218037);
    }
}
