use crate::{_internal::FSCALE32, rng::Rng32, rng32::SplitMix32};
use std::arch::x86_64::*;

#[repr(C, align(64))]
pub struct Sfc32 {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub counter: u32,
}

impl Sfc32 {
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            a: seedgen.nextu(),
            b: seedgen.nextu(),
            c: seedgen.nextu(),
            counter: 1,
        }
    }
}

impl Rng32 for Sfc32 {
    #[inline(always)]
    fn nextu(&mut self) -> u32 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = (self.c << 21) | (self.c >> 11);
        self.c = self.c.wrapping_add(tmp);
        tmp
    }
}

pub(crate) const SFC32X16: usize = 16;

#[repr(C, align(64))]
pub struct Sfc32x16 {
    pub(crate) a: __m512i,
    pub(crate) b: __m512i,
    pub(crate) c: __m512i,
    pub(crate) counter: __m512i,
}

impl Sfc32x16 {
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut a = [0u32; SFC32X16];
        let mut b = [0u32; SFC32X16];
        let mut c = [0u32; SFC32X16];
        for i in 0..SFC32X16 {
            a[i] = seedgen.nextu();
            b[i] = seedgen.nextu();
            c[i] = seedgen.nextu();
        }
        unsafe {
            Self {
                a: _mm512_loadu_si512(a.as_ptr() as *const _),
                b: _mm512_loadu_si512(b.as_ptr() as *const _),
                c: _mm512_loadu_si512(c.as_ptr() as *const _),
                counter: _mm512_set1_epi32(1),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn nextuv(&mut self) -> __m512i {
        let tmp = _mm512_add_epi32(_mm512_add_epi32(self.a, self.b), self.counter);
        self.counter = _mm512_add_epi32(self.counter, _mm512_set1_epi32(1));
        self.a = _mm512_xor_si512(self.b, _mm512_srli_epi32(self.b, 9));
        self.b = _mm512_add_epi32(self.c, _mm512_slli_epi32(self.c, 3));
        self.c = _mm512_add_epi32(_mm512_rol_epi32(self.c, 21), tmp);
        tmp
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn nextfv_scaled(&mut self, scale: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextuv() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_mul_ps(v_f32, scale)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn randiv(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;
        let v_u32 = unsafe { self.nextuv() };
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);
        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);
        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn randfv(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextuv() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_add_ps(_mm512_mul_ps(v_f32, v_mult), v_min)
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; SFC32X16] {
        unsafe {
            let mut result = [0u32; SFC32X16];
            _mm512_storeu_si512(result.as_mut_ptr() as *mut __m512i, self.nextuv());
            result
        }
    }

    pub fn nextf(&mut self) -> [f32; SFC32X16] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{safe_test, unsafe_test};

    safe_test!(Sfc32);
    unsafe_test!(Sfc32x16);
}
