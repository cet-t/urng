use wrapn::{wrap, wu32};

use crate::{Rng, SplitMix32};

/// JSF (Jenkins Small Fast) 32-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::{Rng, Jsf32};
///
/// let mut rng = Jsf32::new(12345);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Jsf32 {
    pub(crate) a: wu32,
    pub(crate) b: wu32,
    pub(crate) c: wu32,
    pub(crate) d: wu32,
}

impl Jsf32 {
    /// Creates a new `Jsf32` instance with the given seed.
    pub const fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            a: wrap!(0xf1ea5eed),
            b: wrap!(seedgen.nextu_const()),
            c: wrap!(seedgen.nextu_const()),
            d: wrap!(seedgen.nextu_const()),
        }
    }
}

impl Rng for Jsf32 {
    type Word = u32;

    #[inline(always)]
    fn nextu(&mut self) -> Self::Word {
        let e = self.a - self.b.rotate_left(27);
        self.a = self.b ^ self.c.rotate_left(17);
        self.b = self.c + self.d;
        self.c = self.d + e;
        self.d = e + self.a;
        *self.d
    }
}

#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "simd")]
pub mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use crate::{Rng, SplitMix32, VRng};

    /// 8-way SIMD implementation of JSF (Jenkins Small Fast) 32-bit RNG.
    /// This implementation uses AVX2 instructions to generate 8 random numbers in parallel.
    ///
    /// # Example
    /// ```no_run
    /// use urng::{VRng, Jsf32x8};
    ///
    /// let mut rng = unsafe { Jsf32x8::new(12345) };
    /// let _ = rng.nextuv();
    /// ```
    #[repr(C, align(64))]
    pub struct Jsf32x8 {
        pub(crate) a: __m256i,
        pub(crate) b: __m256i,
        pub(crate) c: __m256i,
        pub(crate) d: __m256i,
    }

    pub(crate) const JSF32X8: usize = 8;

    impl Jsf32x8 {
        /// # Safety
        #[target_feature(enable = "avx2")]
        pub fn new(seed: u32) -> Self {
            let mut seedgen = SplitMix32::new(seed);
            let mut sv = [[0u32; JSF32X8]; 3];
            for vals in sv.iter_mut() {
                for v in vals.iter_mut() {
                    *v = seedgen.nextu();
                }
            }
            unsafe {
                Self {
                    a: _mm256_set1_epi32(0xf1ea5eed_u32 as i32),
                    b: _mm256_loadu_si256(sv[0].as_ptr() as *const __m256i),
                    c: _mm256_loadu_si256(sv[1].as_ptr() as *const __m256i),
                    d: _mm256_loadu_si256(sv[2].as_ptr() as *const __m256i),
                }
            }
        }
    }

    impl VRng for Jsf32x8 {
        type Word = __m256i;

        #[inline]
        fn nextuv(&mut self) -> __m256i {
            unsafe {
                let e = _mm256_sub_epi32(self.a, _mm256_rol_epi32(self.b, 27));
                self.a = _mm256_xor_si256(self.b, _mm256_rol_epi32(self.c, 17));
                self.b = _mm256_add_epi32(self.c, self.d);
                self.c = _mm256_add_epi32(self.d, e);
                self.d = _mm256_add_epi32(e, self.a);
                self.d
            }
        }
    }

    /// 16-way SIMD implementation of JSF (Jenkins Small Fast) 32-bit RNG.
    /// This implementation uses AVX-512 instructions to generate 16 random numbers in parallel.
    ///
    /// # Example
    /// ```no_run
    /// use urng::{VRng, Jsf32x16};
    ///
    /// unsafe {
    ///     let mut rng = Jsf32x16::new(12345);
    ///     let _ = rng.nextuv();
    /// }
    /// ```
    #[repr(C, align(64))]
    pub struct Jsf32x16 {
        pub(crate) a: __m512i,
        pub(crate) b: __m512i,
        pub(crate) c: __m512i,
        pub(crate) d: __m512i,
    }

    pub(crate) const JSF32X16: usize = 16;

    impl Jsf32x16 {
        /// # Safety
        #[target_feature(enable = "avx512f")]
        pub fn new(seed: u32) -> Self {
            let mut seedgen = SplitMix32::new(seed);
            let mut sv = [[0u32; JSF32X16]; 3];
            for vals in sv.iter_mut() {
                for v in vals.iter_mut() {
                    *v = seedgen.nextu();
                }
            }
            const A: [u32; JSF32X16] = [0xf1ea5eedu32; JSF32X16];
            unsafe {
                Self {
                    a: _mm512_loadu_si512(A.as_ptr() as *const __m512i),
                    b: _mm512_loadu_si512(sv[0].as_ptr() as *const __m512i),
                    c: _mm512_loadu_si512(sv[1].as_ptr() as *const __m512i),
                    d: _mm512_loadu_si512(sv[2].as_ptr() as *const __m512i),
                }
            }
        }
    }

    impl VRng for Jsf32x16 {
        type Word = __m512i;

        fn nextuv(&mut self) -> __m512i {
            unsafe {
                let e = _mm512_sub_epi32(self.a, _mm512_rol_epi32(self.b, 27));
                self.a = _mm512_xor_si512(self.b, _mm512_rol_epi32(self.c, 17));
                self.b = _mm512_add_epi32(self.c, self.d);
                self.c = _mm512_add_epi32(self.d, e);
                self.d = _mm512_add_epi32(e, self.a);
                self.d
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Jsf32);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    crate::unsafe_test!(Jsf32x8);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    crate::unsafe_test!(Jsf32x16);
}
