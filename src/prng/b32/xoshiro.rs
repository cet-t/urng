use wrapn::{wrap, wu32};

use crate::{prng::b32::SplitMix32, rng::Rng};

// --- Xoshiro128++ ---

/// A xoshiro128++ random number generator.
///
/// A fast, high-quality 32-bit generator with a 128-bit state.
/// Uses the ++ scrambler: `rotl(s[0] + s[3], 7) + s[0]`.
///
/// # Example
/// ```
/// use urng::{Rng, Xoshiro128Pp};
///
/// let mut rng = Xoshiro128Pp::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Xoshiro128Pp {
    s: [wu32; 4],
}

impl Xoshiro128Pp {
    /// Creates a new `Xoshiro128Pp` instance with the given seed.
    ///
    /// The seed is expanded via `SplitMix32` to initialize all four state words.
    pub const fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            s: wrap![
                seedgen.nextu_const(),
                seedgen.nextu_const(),
                seedgen.nextu_const(),
                seedgen.nextu_const()
            ],
        }
    }
}

impl Rng for Xoshiro128Pp {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        let res = (self.s[0] + self.s[3]).rotate_left(7) + self.s[0];
        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);

        *res
    }
}

// --- Xoshiro128** ---

/// A xoshiro128** random number generator.
///
/// A fast, high-quality 32-bit generator with a 128-bit state.
/// Uses the ** scrambler: `rotl(s[1] * 5, 7) * 9`.
///
/// # Example
/// ```
/// use urng::{Rng, Xoshiro128Ss};
///
/// let mut rng = Xoshiro128Ss::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Xoshiro128Ss {
    s: [wu32; 4],
}

impl Xoshiro128Ss {
    /// Creates a new `Xoshiro128Ss` instance with the given seed.
    pub const fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            s: wrap![
                seedgen.nextu_const(),
                seedgen.nextu_const(),
                seedgen.nextu_const(),
                seedgen.nextu_const()
            ],
        }
    }
}

impl Rng for Xoshiro128Ss {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        let res = (self.s[1] * 5).rotate_left(7) * 9;
        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(11);

        *res
    }
}

#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "simd")]
pub mod simd {
    use std::arch::x86_64::*;

    use crate::{RngV, SplitMix32};

    // --- Xoshiro128++ x16 ---

    /// 16-way SIMD implementation of xoshiro128++ 32-bit RNG.
    /// This implementation uses AVX-512F instructions to generate 16 random numbers in parallel.
    ///
    /// # Example
    /// ```
    /// use urng::{RngV, Xoshiro128Ppx16};
    ///
    /// unsafe {
    ///     let mut rng = Xoshiro128Ppx16::new(1);
    ///     let _ = rng.nextuv();
    /// }
    /// ```
    #[cfg(target_arch = "x86_64")]
    #[repr(C, align(64))]
    pub struct Xoshiro128Ppx16 {
        s0: __m512i,
        s1: __m512i,
        s2: __m512i,
        s3: __m512i,
    }

    #[cfg(target_arch = "x86_64")]
    impl Xoshiro128Ppx16 {
        /// Creates a new `Xoshiro128Ppx16` instance seeded with the given value.
        ///
        /// # Safety
        ///
        /// Must only be called on a CPU that supports AVX-512F.
        #[target_feature(enable = "avx512f")]
        pub unsafe fn new(seed: u32) -> Self {
            use crate::SplitMix32;

            let mut seedgen = SplitMix32::new(seed);
            let mut sv = [[0u32; 16]; 4];
            for vals in sv.iter_mut() {
                for v in vals {
                    *v = seedgen.nextu_const();
                }
            }
            unsafe {
                Self {
                    s0: _mm512_loadu_si512(sv[0].as_ptr() as *const _),
                    s1: _mm512_loadu_si512(sv[1].as_ptr() as *const _),
                    s2: _mm512_loadu_si512(sv[2].as_ptr() as *const _),
                    s3: _mm512_loadu_si512(sv[3].as_ptr() as *const _),
                }
            }
        }
    }

    impl RngV for Xoshiro128Ppx16 {
        type Word = __m512i;

        #[inline]
        fn nextuv(&mut self) -> Self::Word {
            let s0 = self.s0;
            let s1 = self.s1;
            let s2 = self.s2;
            let s3 = self.s3;

            let s = unsafe {
                let sum = _mm512_add_epi32(s0, s3);
                let rot = _mm512_or_si512(_mm512_slli_epi32(sum, 7), _mm512_srli_epi32(sum, 25));
                let res = _mm512_add_epi32(rot, s0);

                let t = _mm512_slli_epi32(s1, 9);

                let mut s2_next = _mm512_xor_epi32(s2, s0);
                let mut s3_next = _mm512_xor_epi32(s3, s1);
                let s1_next = _mm512_xor_epi32(s1, s2_next);
                let s0_next = _mm512_xor_epi32(s0, s3_next);
                s2_next = _mm512_xor_epi32(s2_next, t);
                s3_next = _mm512_or_si512(
                    _mm512_slli_epi32(s3_next, 11),
                    _mm512_srli_epi32(s3_next, 21),
                );

                [s0_next, s1_next, s2_next, s3_next, res]
            };

            self.s0 = s[0];
            self.s1 = s[1];
            self.s2 = s[2];
            self.s3 = s[3];

            s[4]
        }
    }

    // --- Xoshiro128** x16 ---

    /// 16-way SIMD implementation of xoshiro128** 32-bit RNG.
    /// This implementation uses AVX-512F instructions to generate 16 random numbers in parallel.
    ///
    /// # Example
    /// ```no_run
    /// use urng::{RngV, Xoshiro128Ssx16};
    ///
    /// unsafe {
    ///     let mut rng = Xoshiro128Ssx16::new(1);
    ///     let _ = rng.nextuv();
    /// }
    /// ```
    #[cfg(target_arch = "x86_64")]
    #[repr(C, align(64))]
    pub struct Xoshiro128Ssx16 {
        s0: __m512i,
        s1: __m512i,
        s2: __m512i,
        s3: __m512i,
    }

    #[cfg(target_arch = "x86_64")]
    impl Xoshiro128Ssx16 {
        /// Creates a new `Xoshiro128Ssx16` instance seeded with the given value.
        ///
        /// # Safety
        ///
        /// Must only be called on a CPU that supports AVX-512F.
        #[target_feature(enable = "avx512f")]
        pub unsafe fn new(seed: u32) -> Self {
            let mut seedgen = SplitMix32::new(seed);
            let mut sv = [[0u32; 16]; 4];
            for vals in sv.iter_mut() {
                for v in vals.iter_mut() {
                    *v = seedgen.nextu_const();
                }
            }
            unsafe {
                Self {
                    s0: _mm512_loadu_si512(sv[0].as_ptr() as _),
                    s1: _mm512_loadu_si512(sv[1].as_ptr() as _),
                    s2: _mm512_loadu_si512(sv[2].as_ptr() as _),
                    s3: _mm512_loadu_si512(sv[3].as_ptr() as _),
                }
            }
        }
    }

    impl RngV for Xoshiro128Ssx16 {
        type Word = __m512i;

        #[inline]
        fn nextuv(&mut self) -> Self::Word {
            let s0 = self.s0;
            let s1 = self.s1;
            let s2 = self.s2;
            let s3 = self.s3;

            let s = unsafe {
                let x5 = _mm512_add_epi32(s1, _mm512_slli_epi32(s1, 2));
                let rot = _mm512_or_si512(_mm512_slli_epi32(x5, 7), _mm512_srli_epi32(x5, 25));
                let res = _mm512_add_epi32(rot, _mm512_slli_epi32(rot, 3));

                let mut s2_next = _mm512_xor_epi32(s2, s0);
                let mut s3_next = _mm512_xor_epi32(s3, s1);
                let s1_next = _mm512_xor_epi32(s1, s2_next);
                let s0_next = _mm512_xor_epi32(s0, s3_next);
                s2_next = _mm512_xor_epi32(s2_next, _mm512_slli_epi32(s1, 9));
                s3_next = _mm512_or_si512(
                    _mm512_slli_epi32(s3_next, 11),
                    _mm512_srli_epi32(s3_next, 21),
                );

                [s0_next, s1_next, s2_next, s3_next, res]
            };

            self.s0 = s[0];
            self.s1 = s[1];
            self.s2 = s[2];
            self.s3 = s[3];

            s[4]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! {
        Xoshiro128Pp,
        Xoshiro128Ss
    }
}
