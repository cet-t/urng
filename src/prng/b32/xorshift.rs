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
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
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
        *self.a
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
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
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
        *self.x[0]
    }
}

#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "simd")]
pub mod simd {
    use std::arch::x86_64::*;

    use crate::{RngV, SplitMix32};

    const XORSHIFT32X8: usize = 8;

    #[repr(C, align(64))]
    pub struct Xorshift32x8 {
        a: __m256i,
    }

    impl Xorshift32x8 {
        pub fn new(seed: u32) -> Self {
            let mut sm = SplitMix32::new(seed);

            let mut a = [0u32; XORSHIFT32X8];
            for i in 0..XORSHIFT32X8 {
                a[i] = sm.nextu_const();
            }

            unsafe {
                Self {
                    a: _mm256_loadu_epi32(a.as_ptr() as _),
                }
            }
        }
    }

    impl RngV for Xorshift32x8 {
        type Word = __m256i;

        #[inline]
        fn nextuv(&mut self) -> Self::Word {
            unsafe {
                let x = self.a;

                self.a = _mm256_xor_si256(x, _mm256_slli_epi32::<13>(x));
                self.a = _mm256_xor_si256(self.a, _mm256_srli_epi32::<17>(self.a));
                self.a = _mm256_xor_si256(self.a, _mm256_slli_epi32::<5>(self.a));
            }
            self.a
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! {
        Xorshift32,
        Xorshift128
    }
}
