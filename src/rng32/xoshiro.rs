use std::{
    arch::x86_64::{
        __m512, __m512i, _mm512_add_epi32, _mm512_add_ps, _mm512_cvtepu32_ps, _mm512_loadu_si512,
        _mm512_mask_blend_epi32, _mm512_mul_epu32, _mm512_mul_ps, _mm512_or_si512,
        _mm512_set1_epi32, _mm512_set1_epi64, _mm512_set1_ps, _mm512_slli_epi32, _mm512_srli_epi32,
        _mm512_srli_epi64, _mm512_storeu_ps, _mm512_xor_epi32,
    },
    num::Wrapping,
};

use crate::{rng::Rng32, rng32::SplitMix32, wrap};

// --- Xoshiro128++ ---

/// A xoshiro128++ random number generator.
///
/// A fast, high-quality 32-bit generator with a 128-bit state.
/// Uses the ++ scrambler: `rotl(s[0] + s[3], 7) + s[0]`.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xoshiro128Pp::new(1);
/// assert_eq!(rng.nextu(), 4075539671);
/// ```
#[repr(C)]
pub struct Xoshiro128Pp {
    s: [Wrapping<u32>; 4],
}

impl Xoshiro128Pp {
    /// Creates a new `Xoshiro128Pp` instance seeded with the given value.
    ///
    /// The seed is expanded via `SplitMix32` to initialize all four state words.
    ///
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            s: wrap![
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu()
            ],
        }
    }
}

impl Rng32 for Xoshiro128Pp {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let res = wrap!((self.s[0] + self.s[3]).0.rotate_left(7)) + self.s[0];
        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = wrap!(self.s[3].0.rotate_left(11));

        res.0
    }
}

// --- Xoshiro128** ---

/// A xoshiro128** random number generator.
///
/// A fast, high-quality 32-bit generator with a 128-bit state.
/// Uses the ** scrambler: `rotl(s[1] * 5, 7) * 9`.
///
/// # Examples
///
/// ```
/// use urng::rng::Rng32;
/// use urng::rng32::Xoshiro128Ss;
///
/// let mut rng = Xoshiro128Ss::new(1);
/// assert_eq!(rng.nextu(), 997331382);
/// ```
#[repr(C)]
pub struct Xoshiro128Ss {
    s: [Wrapping<u32>; 4],
}

impl Xoshiro128Ss {
    /// Creates a new `Xoshiro128Ss` instance seeded with the given value.
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            s: wrap![
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu()
            ],
        }
    }
}

impl Rng32 for Xoshiro128Ss {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let res = wrap!((self.s[1] * wrap!(5)).0.rotate_left(7)) * wrap!(9);
        let t = self.s[1] << 9;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = wrap!(self.s[3].0.rotate_left(11));

        res.0
    }
}

// --- Xoshiro128++ x16 ---

/// A xoshiro128++ random number generator using AVX-512 to process 16 lanes simultaneously.
///
/// State: four `__m512i` registers, each holding 16 × u32 values for s[0]..s[3].
/// Requires AVX-512F support.
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
        let mut seedgen = SplitMix32::new(seed);
        let mut sv = [[0u32; 16]; 4];
        for vals in sv.iter_mut() {
            for v in vals {
                *v = seedgen.nextu();
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

    /// Generates the next 16 random `u32` values as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu_vec(&mut self) -> __m512i {
        let s0 = self.s0;
        let s1 = self.s1;
        let s2 = self.s2;
        let s3 = self.s3;

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

        self.s0 = s0_next;
        self.s1 = s1_next;
        self.s2 = s2_next;
        self.s3 = s3_next;

        res
    }

    /// Generates the next 16 random `f32` values in the range [0, 1) as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf_vec_scaled(&mut self, scale: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextu_vec() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_mul_ps(v_f32, scale)
    }

    /// Generates the next 16 random `i32` values in the range [min, max] as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi_vec(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;

        let v_u32 = unsafe { self.nextu_vec() };
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);

        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    /// Generates the next 16 random `f32` values in the range [min, max) as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf_vec(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextu_vec() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_add_ps(_mm512_mul_ps(v_f32, v_mult), v_min)
    }

    /// Generates the next 16 random `u32` values.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; 16] {
        unsafe { std::mem::transmute(self.nextu_vec()) }
    }

    /// Generates 16 random `f32` values in the range [0, 1).
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf(&mut self) -> [f32; 16] {
        let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
        unsafe { std::mem::transmute(self.nextf_vec_scaled(scale)) }
    }

    /// Generates 16 random `i32` values in the range [min, max].
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi(&mut self, min: i32, max: i32) -> [i32; 16] {
        let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
        let v_min = _mm512_set1_epi32(min);
        unsafe { std::mem::transmute(self.randi_vec(v_range, v_min)) }
    }

    /// Generates 16 random `f32` values in the range [min, max).
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf(&mut self, min: f32, max: f32) -> [f32; 16] {
        let v_mult = _mm512_set1_ps((max - min) * (1.0 / (u32::MAX as f32 + 1.0)));
        let v_min = _mm512_set1_ps(min);
        unsafe { std::mem::transmute(self.randf_vec(v_mult, v_min)) }
    }
}

// --- Xoshiro128** x16 ---

/// A xoshiro128** random number generator using AVX-512 to process 16 lanes simultaneously.
///
/// State: four `__m512i` registers, each holding 16 × u32 values for s[0]..s[3].
/// Requires AVX-512F support.
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
                *v = seedgen.nextu();
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

    /// Generates the next 16 random `u32` values.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu_vec(&mut self) -> __m512i {
        let s0 = self.s0;
        let s1 = self.s1;
        let s2 = self.s2;
        let s3 = self.s3;

        // res = rotl(s1 * 5, 7) * 9, with shift-add instead of mul by constants.
        let x5 = _mm512_add_epi32(s1, _mm512_slli_epi32(s1, 2));
        let rot = _mm512_or_si512(_mm512_slli_epi32(x5, 7), _mm512_srli_epi32(x5, 25));
        let res = _mm512_add_epi32(rot, _mm512_slli_epi32(rot, 3));

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

        self.s0 = s0_next;
        self.s1 = s1_next;
        self.s2 = s2_next;
        self.s3 = s3_next;

        res
    }

    /// Generates the next 16 random `f32` values in the range [0, 1) as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf_vec_scaled(&mut self, scale: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextu_vec() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_mul_ps(v_f32, scale)
    }

    /// Generates the next 16 random `i32` values in the range [min, max] as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi_vec(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;

        let v_u32 = unsafe { self.nextu_vec() };
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);

        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    /// Generates the next 16 random `f32` values in the range [min, max) as a vector register.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf_vec(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let v_u32 = unsafe { self.nextu_vec() };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        _mm512_add_ps(_mm512_mul_ps(v_f32, v_mult), v_min)
    }

    /// Generates the next 16 random `u32` values.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; 16] {
        unsafe { std::mem::transmute(self.nextu_vec()) }
    }

    /// Generates 16 random `f32` values in the range [0, 1).
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf(&mut self) -> [f32; 16] {
        let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
        unsafe { std::mem::transmute(self.nextf_vec_scaled(scale)) }
    }

    /// Generates 16 random `i32` values in the range [min, max].
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi(&mut self, min: i32, max: i32) -> [i32; 16] {
        let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
        let v_min = _mm512_set1_epi32(min);
        unsafe { std::mem::transmute(self.randi_vec(v_range, v_min)) }
    }

    /// Generates 16 random `f32` values in the range [min, max).
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf(&mut self, min: f32, max: f32) -> [f32; 16] {
        let v_mult = _mm512_set1_ps((max - min) * (1.0 / (u32::MAX as f32 + 1.0)));
        let v_min = _mm512_set1_ps(min);
        let mut out = [0f32; 16];
        unsafe { _mm512_storeu_ps(out.as_mut_ptr(), self.randf_vec(v_mult, v_min)) };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xoshiro128Pp);
    crate::safe_test!(Xoshiro128Ss);
}
