use crate::{_internal::FSCALE32, rng::Rng32, rng32::SplitMix32};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// JSF (Jenkins Small Fast) 32-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::rng::Rng32;
/// use urng::rng32::Jsf32;
///
/// let mut rng = Jsf32::new(12345);
/// ```
#[repr(C, align(64))]
pub struct Jsf32 {
    pub(crate) a: u32,
    pub(crate) b: u32,
    pub(crate) c: u32,
    pub(crate) d: u32,
}

impl Jsf32 {
    /// Creates a new `Jsf32` instance with the given seed.
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            a: 0xf1ea5eed,
            b: seedgen.nextu(),
            c: seedgen.nextu(),
            d: seedgen.nextu(),
        }
    }
}

impl Rng32 for Jsf32 {
    #[inline(always)]
    fn nextu(&mut self) -> u32 {
        let e = self.a.wrapping_sub(self.b.rotate_left(27));
        self.a = self.b ^ self.c.rotate_left(17);
        self.b = self.c.wrapping_add(self.d);
        self.c = self.d.wrapping_add(e);
        self.d = e.wrapping_add(self.a);
        self.d
    }
}

/// 16-way SIMD implementation of JSF (Jenkins Small Fast) 32-bit RNG.
/// This implementation uses AVX-512 instructions to generate 16 random numbers in parallel.
///
/// # Example
/// ```
/// use urng::rng::Rng32;
/// use urng::rng32::Jsf32x16;
///
/// let mut rng = unsafe { Jsf32x16::new(12345) };
/// let _ = unsafe { rng.nextu() };
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
        let a = [0xf1ea5eedu32; JSF32X16];
        unsafe {
            Self {
                a: _mm512_loadu_si512(a.as_ptr() as *const __m512i),
                b: _mm512_loadu_si512(sv[0].as_ptr() as *const __m512i),
                c: _mm512_loadu_si512(sv[1].as_ptr() as *const __m512i),
                d: _mm512_loadu_si512(sv[2].as_ptr() as *const __m512i),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn nextu_vec(&mut self) -> __m512i {
        let e = _mm512_sub_epi32(self.a, _mm512_rol_epi32(self.b, 27));
        self.a = _mm512_xor_si512(self.b, _mm512_rol_epi32(self.c, 17));
        self.b = _mm512_add_epi32(self.c, self.d);
        self.c = _mm512_add_epi32(self.d, e);
        self.d = _mm512_add_epi32(e, self.a);
        self.d
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn nextf_vec_scaled(&mut self, scale: __m512) -> __m512 {
        let v_u32 = self.nextu_vec();
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_mul_ps(v_f32, scale)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn randi_vec(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;
        let v_u32 = self.nextu_vec();
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);
        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);
        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn randf_vec(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let v_u32 = self.nextu_vec();
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_add_ps(_mm512_mul_ps(v_f32, v_mult), v_min)
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; JSF32X16] {
        unsafe {
            let mut result = [0u32; JSF32X16];
            _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, self.nextu_vec());
            result
        }
    }

    #[inline(always)]
    pub fn nextf(&mut self) -> [f32; JSF32X16] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{safe_test, unsafe_test};

    safe_test!(Jsf32);
    unsafe_test!(Jsf32x16);
}
