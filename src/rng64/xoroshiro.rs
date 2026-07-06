use wrapn::{Wrap, wrap};

use crate::{rng::Rng64, rng64::SplitMix64};

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
    s: [Wrap<u64>; 2],
}

impl Xoroshiro128Pp {
    /// Creates a new `Xoroshiro128Pp` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Self {
            s: wrap![seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Applies the jump function for advancing by $2^{64}$ steps.
    #[inline]
    pub fn jump(&mut self) {
        const JUMP: [u64; 2] = [0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05];

        macro_rules! jump {
            ($i:expr, $s:ident) => {
                // TODO: unrolling
                for b in 0..64 {
                    if (JUMP[$i] & (1 << b)) != 0 {
                        $s[0] ^= self.s[0];
                        $s[1] ^= self.s[1];
                    }
                    self.nextu();
                }
            };
        }

        let mut s = wrap![0; 2];
        jump!(0, s);
        jump!(1, s);
        self.s = s;
    }

    /// Applies the long-jump function for advancing by $2^{96}$ steps.
    #[inline]
    pub fn long_jump(&mut self) {
        const LONG_JUMP: [u64; 2] = [0x360fd5f2cf8d5d99, 0x9c6e6877736c46e3];

        macro_rules! long_jump {
            ($i:expr, $s:ident) => {
                for i in 0..LONG_JUMP.len() {
                    // TODO: unrolling
                    for b in 0..64 {
                        if (LONG_JUMP[i] & (1 << b)) != 0 {
                            $s[0] ^= self.s[0];
                            $s[1] ^= self.s[1];
                        }
                        self.nextu();
                    }

                    self.s = $s;
                }
            };
        }

        let mut s = wrap![0; 2];
        long_jump!(0, s);
        long_jump!(1, s);
        self.s = s;
    }
}

impl Rng64 for Xoroshiro128Pp {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let s = self.s;
        let result = s[0] + s[1];

        self.s[0] = s[1] ^ s[0];
        self.s[1] = self.s[1] ^ s[0].rotate_left(24) ^ (self.s[1] << 16);
        self.s[1] = self.s[1].rotate_left(37);

        *result.raw()
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
    s: [Wrap<u64>; 2],
}

impl Xoroshiro128Ss {
    /// Creates a new `Xoroshiro128Ss` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: wrap![seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Applies the jump function for advancing by $2^{64}$ steps.
    #[inline]
    pub fn jump(&mut self) {
        const JUMP: [u64; 2] = [0xdf900294d8f554a5, 0x170865df4b3201fc];

        macro_rules! jump {
            ($i:expr, $s:ident) => {
                // TODO: unrolling
                for b in 0..64 {
                    if (JUMP[$i] & (1 << b)) != 0 {
                        $s[0] ^= self.s[0];
                        $s[1] ^= self.s[1];
                    }
                    self.nextu();
                }
            };
        }

        let mut s = wrap![0; 2];
        jump!(0, s);
        jump!(1, s);
        self.s = s;
    }

    /// Applies the long-jump function for advancing by $2^{96}$ steps.
    #[inline]
    pub fn long_jump(&mut self) {
        const LONG_JUMP: [u64; 2] = [0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1];

        macro_rules! long_jump {
            ($i:expr, $s:ident) => {
                for i in 0..LONG_JUMP.len() {
                    for b in 0..64 {
                        if (LONG_JUMP[i] & (1 << b)) != 0 {
                            $s[0] ^= self.s[0];
                            $s[1] ^= self.s[1];
                        }
                        self.nextu();
                    }

                    self.s = $s;
                }
            };
        }

        let mut s = wrap![0; 2];
        long_jump!(0, s);
        long_jump!(1, s);
        self.s = s;
    }
}

impl Rng64 for Xoroshiro128Ss {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let result = (self.s[0] * 5).rotate_left(7) * 9;

        self.s[1] = self.s[1] ^ self.s[0];
        self.s[0] = self.s[1] ^ self.s[0].rotate_left(24) ^ (self.s[1] << 16);
        self.s[1] = self.s[1].rotate_left(37);

        result.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xoroshiro128Pp);
    crate::safe_test!(Xoroshiro128Ss);
}
