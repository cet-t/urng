#[cfg(feature = "simd")]
use std::arch::x86_64::*;

use crate::_internal::{FSCALE32, FSCALE64};

macro_rules! randi_wide {
    (i 32) => {
        i64
    };
    (i 64) => {
        i128
    };
    (u 32) => {
        u64
    };
    (u 64) => {
        u128
    };
}

macro_rules! impl_rng_trait {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("A trait for ", $bits, "-bit random number generators.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use urng::rng::Rng", $bits, ";")]
            #[doc = concat!("use urng::rng::Choice", $bits, ";")]
            #[doc = concat!("use urng::rng", $bits, "::Xorshift", $bits, ";")]
            ///
            #[doc = concat!("let mut rng = Xorshift", $bits, "::new(1);")]
            #[doc = concat!("let val = rng.randi(1, 6);")]
            #[doc = concat!("assert!((1..=6).contains(&val));")]
            #[doc = concat!("assert!(rng.randf(0.0, 1.0_f", $bits, ") < 1.0);")]
            #[doc = concat!("let items = [\"a\", \"b\", \"c\"];")]
            #[doc = concat!("assert!(items.contains(rng.choice(&items)));")]
            /// ```
            pub trait [<Rng $bits>] {
                #[doc = concat!("Generates the next random `u", $bits, "` value in the range [0, 2^", $bits, ").")]
                fn nextu(&mut self) -> [<u $bits>];

                #[doc = concat!("Generates the next random `f", $bits, "` value in the range [0, 1).")]
                #[inline(always)]
                fn nextf(&mut self) -> [<f $bits>] {
                    self.nextu() as [<f $bits>] * [<FSCALE $bits>]
                }

                #[doc = concat!("Generates a random `i", $bits, "` value in the range [min, max].")]
                #[inline(always)]
                fn randi(&mut self, min: [<i $bits>], max: [<i $bits>]) -> [<i $bits>] {
                    let range = (max as randi_wide!(i $bits) - min as randi_wide!(i $bits) + 1)
                        as randi_wide!(u $bits);
                    ((self.nextu() as randi_wide!(u $bits) * range) >> $bits) as [<i $bits>] + min
                }

                #[doc = concat!("Generates a random `f", $bits, "` value in the range [min, max).")]
                #[inline(always)]
                fn randf(&mut self, min: [<f $bits>], max: [<f $bits>]) -> [<f $bits>] {
                    let scale = (max - min) * [<FSCALE $bits>];
                    (self.nextu() as [<f $bits>] * scale) + min
                }
            }

            pub trait [<Choice $bits>]: [<Rng $bits>] {
                /// Returns a random element from a slice.
                #[inline(always)]
                fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
                    let index = self.randi(0, choices.len() as [<i $bits>] - 1);
                    &choices[index as usize]
                }
            }

            impl<T: [<Rng $bits>] + ?Sized> [<Choice $bits>] for T {}
        }
    };
}

impl_rng_trait!(32);
impl_rng_trait!(64);

#[cfg(feature = "simd")]
pub trait Rng32V256 {
    /// Generates the next random `__m256i` value containing 8 `u32` integers in the range [0, 2^32).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX2 instructions. The caller must check for AVX2 support before calling this function to avoid undefined behavior.
    unsafe fn nextuv(&mut self) -> __m256i;

    /// Generates the next random `__m256` value containing 8 `f32` floats in the range [0.0, 1.0).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX2 instructions. The caller must check for AVX2 support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn nextfv(&mut self, v_scale: __m256) -> __m256 {
        let uv = unsafe { self.nextuv() };
        let fv = _mm256_cvtepi32_ps(uv);
        _mm256_mul_ps(fv, v_scale)
    }

    /// Generates a random `__m256i` value containing 8 `i32` integers in the range [v_min, v_min + v_range).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX2 instructions. The caller must check for AVX2 support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn randiv(&mut self, v_range: __m256i, v_min: __m256i) -> __m256i {
        const MERGE_MASK: u8 = 0xAA;

        let uv = unsafe { self.nextuv() };
        let prod_even = _mm256_mul_epu32(uv, v_range);
        let res_even = _mm256_srli_epi64(prod_even, 32);
        let v_u32_shifted = _mm256_srli_epi64(uv, 32);
        let prod_odd = _mm256_mul_epu32(v_u32_shifted, v_range);
        let merged = unsafe { _mm256_mask_blend_epi32(MERGE_MASK, res_even, prod_odd) };
        _mm256_add_epi32(merged, v_min)
    }

    /// Generates a random `__m256` value containing 8 `f32` floats in the range [v_min, v_min + v_mult).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX2 instructions. The caller must check for AVX2 support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn randfv(&mut self, v_mult: __m256, v_min: __m256) -> __m256 {
        let uv = unsafe { self.nextuv() };
        let fv = _mm256_cvtepi32_ps(uv);
        _mm256_add_ps(_mm256_mul_ps(fv, v_mult), v_min)
    }

    /// Generates the next random `u32` integers in the range [0, 2^32) and returns them as an array of 8 elements.
    #[inline(always)]
    fn nextu(&mut self) -> [u32; 8] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    /// Generates the next random `f32` floats in the range [0.0, 1.0) and returns them as an array of 8 elements.
    #[inline(always)]
    fn nextf(&mut self) -> [f32; 8] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

#[cfg(feature = "simd")]
pub trait Rng32V512 {
    /// Generates the next random `__m512i` value containing 16 `u32` integers in the range [0, 2^32).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX-512F instructions. The caller must check for AVX-512F support before calling this function to avoid undefined behavior.
    unsafe fn nextuv(&mut self) -> __m512i;

    /// Generates the next random `__m512` value containing 16 `f32` floats in the range [0.0, 1.0).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX-512F instructions. The caller must check for AVX-512F support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn nextfv(&mut self, v_scale: __m512) -> __m512 {
        let uv = unsafe { self.nextuv() };
        let fv = _mm512_cvtepu32_ps(uv);
        _mm512_mul_ps(fv, v_scale)
    }

    /// Generates a random `__m512i` value containing 16 `i32` integers in the range [v_min, v_min + v_range).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX-512F instructions. The caller must check for AVX-512F support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn randiv(&mut self, v_range: __m512i, v_min: __m512i) -> __m512i {
        const MERGE_MASK: u16 = 0xAAAA;
        let uv = unsafe { self.nextuv() };
        let prod_even = _mm512_mul_epu32(uv, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);
        let v_u32_shifted = _mm512_srli_epi64(uv, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);
        let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
        _mm512_add_epi32(merged, v_min)
    }

    /// Generates a random `__m512` value containing 16 `f32` floats in the range [v_min, v_min + v_mult).
    ///
    /// # Safety
    ///
    /// This function is unsafe because it relies on the caller to ensure that the CPU supports AVX-512F instructions. The caller must check for AVX-512F support before calling this function to avoid undefined behavior.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn randfv(&mut self, v_mult: __m512, v_min: __m512) -> __m512 {
        let uv = unsafe { self.nextuv() };
        let fv = _mm512_cvtepu32_ps(uv);
        _mm512_add_ps(_mm512_mul_ps(fv, v_mult), v_min)
    }

    /// Generates the next random `u32` integers in the range [0, 2^32) and returns them as an array of 16 elements.
    #[inline(always)]
    fn nextu(&mut self) -> [u32; 16] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    /// Generates the next random `f32` floats in the range [0.0, 1.0) and returns them as an array of 16 elements.
    #[inline(always)]
    fn nextf(&mut self) -> [f32; 16] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}
