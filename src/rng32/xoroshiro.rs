use crate::_internal::FSCALE32;
use crate::rng::Rng32;
use crate::rng32::SplitMix32;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct Xoroshiro64Ss {
    s: [u32; 2],
}

impl Xoroshiro64Ss {
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);

        Self {
            s: [seedgen.nextu(), seedgen.nextu()],
        }
    }
}

impl Rng32 for Xoroshiro64Ss {
    #[inline(always)]
    fn nextu(&mut self) -> u32 {
        let s0 = self.s[0];
        let mut s1 = self.s[1];
        let result = s0.wrapping_mul(0x9E3779BB).rotate_left(5).wrapping_mul(5);

        s1 ^= s0;
        self.s[0] = s0.rotate_left(26) ^ s1 ^ (s1 << 9);
        self.s[1] = s1.rotate_left(13);

        result
    }
}

pub(crate) const XOROSHIRO64SSX8: usize = 8;

#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Xoroshiro64Ssx8 {
    s0: __m256i,
    s1: __m256i,
}

impl Xoroshiro64Ssx8 {
    /// Initializes the generator with a given seed, filling the state arrays with values derived from the seed.
    ///
    /// # Safety
    /// This function requires AVX2 support. Ensure that the CPU supports it and that the code is compiled with the appropriate target features.
    #[target_feature(enable = "avx2")]
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);

        let mut s0 = [0u32; XOROSHIRO64SSX8];
        let mut s1 = [0u32; XOROSHIRO64SSX8];

        for i in 0..XOROSHIRO64SSX8 {
            s0[i] = seedgen.nextu();
            s1[i] = seedgen.nextu();
        }

        unsafe {
            Self {
                s0: _mm256_loadu_si256(s0.as_ptr() as *const __m256i),
                s1: _mm256_loadu_si256(s1.as_ptr() as *const __m256i),
            }
        }
    }

    /// Generates the next 8 random numbers in parallel.
    ///
    /// # Safety
    /// This function requires AVX2 support. Ensure that the CPU supports it and that the code is compiled with the appropriate target features.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn nextuv(&mut self) -> __m256i {
        let s0 = self.s0;
        let mut s1 = self.s1;

        let mult = _mm256_set1_epi32(0x9E3779BBu32 as i32);
        let result = _mm256_mullo_epi32(s0, mult);

        s1 = _mm256_xor_si256(s1, s0);
        self.s0 = _mm256_xor_si256(
            _mm256_xor_si256(unsafe { _mm256_rol_epi32(s0, 26) }, s1),
            _mm256_slli_epi32(s1, 9),
        );
        self.s1 = unsafe { _mm256_rol_epi32(s1, 13) };

        result
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; XOROSHIRO64SSX8] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn nextfv(&mut self, scale: __m256) -> __m256 {
        let v_f32 = _mm256_cvtepi32_ps(self.nextuv());
        _mm256_mul_ps(v_f32, scale)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn randiv(&mut self, v_range: __m256i, v_min: __m256i) -> __m256i {
        let v = self.nextuv();
        let res_even = _mm256_srli_epi64(_mm256_mul_epu32(v, v_range), 32);
        let v_hi = _mm256_srli_epi64(v, 32);
        let prod_odd = _mm256_slli_epi64(
            _mm256_srli_epi64(_mm256_mul_epu32(v_hi, v_range), 32),
            32,
        );
        _mm256_add_epi32(_mm256_or_si256(res_even, prod_odd), v_min)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) fn randfv(&mut self, v_mult: __m256, v_min: __m256) -> __m256 {
        let v_f32 = _mm256_cvtepi32_ps(self.nextuv());
        _mm256_add_ps(_mm256_mul_ps(v_f32, v_mult), v_min)
    }
}

pub(crate) const XOROSHIRO64SSX16: usize = 16;

#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Xoroshiro64Ssx16 {
    s0: __m512i,
    s1: __m512i,
}

impl Xoroshiro64Ssx16 {
    /// Initializes the generator with a given seed, filling the state arrays with values derived from the seed.
    ///
    /// # Safety
    /// This function requires AVX-512F support. Ensure that the CPU supports it and that the code is compiled with the appropriate target features.
    #[target_feature(enable = "avx512f")]
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);

        let mut s0 = [0u32; XOROSHIRO64SSX16];
        let mut s1 = [0u32; XOROSHIRO64SSX16];

        for i in 0..XOROSHIRO64SSX16 {
            s0[i] = seedgen.nextu();
            s1[i] = seedgen.nextu();
        }

        unsafe {
            Self {
                s0: _mm512_loadu_si512(s0.as_ptr() as *const __m512i),
                s1: _mm512_loadu_si512(s1.as_ptr() as *const __m512i),
            }
        }
    }

    /// Generates the next 16 random numbers in parallel.
    ///
    /// # Safety
    /// This function requires AVX-512F support. Ensure that the CPU supports it and that the code is compiled with the appropriate target features.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn nextuv(&mut self) -> __m512i {
        let s0 = self.s0;
        let mut s1 = self.s1;

        let mult = _mm512_set1_epi32(0x9E3779BBu32 as i32);
        let result = _mm512_mullo_epi32(s0, mult);

        s1 = _mm512_xor_si512(s1, s0);
        self.s0 = _mm512_xor_si512(
            _mm512_xor_si512(_mm512_rol_epi32(s0, 26), s1),
            _mm512_slli_epi32(s1, 9),
        );
        self.s1 = _mm512_rol_epi32(s1, 13);

        result
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; XOROSHIRO64SSX16] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    pub fn nextf(&mut self) -> [f32; XOROSHIRO64SSX16] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn nextfv(&mut self, scale: __m512) -> __m512 {
        let v_f32 = _mm512_cvtepu32_ps(self.nextuv());
        _mm512_mul_ps(v_f32, scale)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn randiv(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;
        let v = self.nextuv();
        let res_even = _mm512_srli_epi64(_mm512_mul_epu32(v, v_range), 32);
        let prod_odd = _mm512_mul_epu32(_mm512_srli_epi64(v, 32), v_range);
        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) fn randfv(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let v_f32 = _mm512_cvtepu32_ps(self.nextuv());
        _mm512_add_ps(_mm512_mul_ps(v_f32, v_mult), v_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{safe_test, unsafe_test};

    safe_test!(Xoroshiro64Ss);
    unsafe_test!(Xoroshiro64Ssx16);
}
