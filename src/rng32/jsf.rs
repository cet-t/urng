#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use wrapn::Wrap;

use crate::{_internal::impl_seed, Rng, SplitMix32};
#[cfg(feature = "simd")]
use crate::{Rng32V256, Rng32V512};

/// JSF (Jenkins Small Fast) 32-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::rng::Rng;
/// use urng::rng32::Jsf32;
///
/// let mut rng = Jsf32::new(12345);
/// ```
#[repr(C, align(64))]
pub struct Jsf32 {
    pub(crate) a: Wrap<u32>,
    pub(crate) b: Wrap<u32>,
    pub(crate) c: Wrap<u32>,
    pub(crate) d: Wrap<u32>,
}

impl Jsf32 {
    /// Creates a new `Jsf32` instance with the given seed.
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            a: 0xf1ea5eed.into(),
            b: seedgen.nextu().into(),
            c: seedgen.nextu().into(),
            d: seedgen.nextu().into(),
        }
    }
}

impl_seed!(Jsf32, 32);

impl Rng for Jsf32 {
    type Word = u32;
    #[inline(always)]
    fn nextu(&mut self) -> u32 {
        let e = self.a - self.b.rotate_left(27);
        self.a = self.b ^ self.c.rotate_left(17);
        self.b = self.c + self.d;
        self.c = self.d + e;
        self.d = e + self.a;
        self.d.value()
    }
}

#[cfg(feature = "simd")]
#[repr(C, align(64))]
pub struct Jsf32x8 {
    pub(crate) a: __m256i,
    pub(crate) b: __m256i,
    pub(crate) c: __m256i,
    pub(crate) d: __m256i,
}

#[cfg(feature = "simd")]
pub(crate) const JSF32X8: usize = 8;

#[cfg(feature = "simd")]
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
        let a = [0xf1ea5eedu32; JSF32X8];
        unsafe {
            Self {
                a: _mm256_loadu_si256(a.as_ptr() as *const __m256i),
                b: _mm256_loadu_si256(sv[0].as_ptr() as *const __m256i),
                c: _mm256_loadu_si256(sv[1].as_ptr() as *const __m256i),
                d: _mm256_loadu_si256(sv[2].as_ptr() as *const __m256i),
            }
        }
    }
}

#[cfg(feature = "simd")]
impl Rng32V256 for Jsf32x8 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn nextuv(&mut self) -> __m256i {
        let e = _mm256_sub_epi32(self.a, unsafe { _mm256_rol_epi32(self.b, 27) });
        self.a = _mm256_xor_si256(self.b, unsafe { _mm256_rol_epi32(self.c, 17) });
        self.b = _mm256_add_epi32(self.c, self.d);
        self.c = _mm256_add_epi32(self.d, e);
        self.d = _mm256_add_epi32(e, self.a);
        self.d
    }

    #[inline(always)]
    fn nextu(&mut self) -> [u32; JSF32X8] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }
}

/// 16-way SIMD implementation of JSF (Jenkins Small Fast) 32-bit RNG.
/// This implementation uses AVX-512 instructions to generate 16 random numbers in parallel.
///
/// # Example
/// ```no_run
/// use urng::Rng32V512;
/// use urng::rng32::Jsf32x16;
///
/// let mut rng = unsafe { Jsf32x16::new(12345) };
/// let _ = unsafe { rng.nextu() };
/// ```
#[cfg(feature = "simd")]
#[repr(C, align(64))]
pub struct Jsf32x16 {
    pub(crate) a: __m512i,
    pub(crate) b: __m512i,
    pub(crate) c: __m512i,
    pub(crate) d: __m512i,
}

#[cfg(feature = "simd")]
pub(crate) const JSF32X16: usize = 16;

#[cfg(feature = "simd")]
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

#[cfg(feature = "simd")]
impl Rng32V512 for Jsf32x16 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn nextuv(&mut self) -> __m512i {
        let e = _mm512_sub_epi32(self.a, _mm512_rol_epi32(self.b, 27));
        self.a = _mm512_xor_si512(self.b, _mm512_rol_epi32(self.c, 17));
        self.b = _mm512_add_epi32(self.c, self.d);
        self.c = _mm512_add_epi32(self.d, e);
        self.d = _mm512_add_epi32(e, self.a);
        self.d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe_test;
    #[cfg(all(
        feature = "simd",
        any(target_feature = "avx2", target_feature = "avx512f")
    ))]
    use crate::unsafe_test;

    safe_test!(Jsf32);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    unsafe_test!(Jsf32x8);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    unsafe_test!(Jsf32x16);
}
