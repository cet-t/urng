use std::arch::x86_64::*;

use crate::_internal::simd_f01;

mod sealed {
    use std::arch::x86_64::{__m128i, __m256i, __m512i};

    pub trait Sealed {}
    impl Sealed for __m128i {}
    impl Sealed for __m256i {}
    impl Sealed for __m512i {}
}

pub trait WordV: sealed::Sealed + Copy + Sized {
    type Int;
    type Float;

    #[must_use]
    fn to_f01(self) -> Self::Float;

    #[must_use]
    fn to_randi(self, scale: Self::Int, min: Self::Int) -> Self::Int;

    #[must_use]
    fn to_randf(self, scale: Self::Float, min: Self::Float) -> Self::Float;
}

impl WordV for __m128i {
    type Int = Self;
    type Float = __m128;

    fn to_f01(self) -> Self::Float {
        unsafe { simd_f01::u32x4(self) }
    }

    fn to_randi(self, scale: Self::Int, min: Self::Int) -> Self::Int {
        const MERGE_MASK: i32 = 0b10001000;

        unsafe {
            let prod_even = _mm_mul_epu32(self, scale);
            let res_even = _mm_srli_epi64(prod_even, 32);
            let v_u32_shifted = _mm_srli_epi64(self, 32);
            let prod_odd = _mm_mul_epu32(v_u32_shifted, scale);
            let merged = _mm_castps_si128(_mm_shuffle_ps(
                _mm_castsi128_ps(res_even),
                _mm_castsi128_ps(prod_odd),
                MERGE_MASK,
            ));
            let merged = _mm_shuffle_epi32(merged, 0b11_01_10_00);
            _mm_add_epi32(merged, min)
        }
    }

    fn to_randf(self, scale: Self::Float, min: Self::Float) -> Self::Float {
        unsafe {
            _mm_add_ps(_mm_mul_ps(simd_f01::u32x4(self), scale), min)
        }
    }
}

impl WordV for __m256i {
    type Int = Self;
    type Float = __m256;

    fn to_f01(self) -> Self::Float {
        unsafe { simd_f01::u32x8(self) }
    }

    fn to_randi(self, scale: Self::Int, min: Self::Int) -> Self::Int {
        const MERGE_MASK: u8 = 0b10101010;

        unsafe {
            let prod_even = _mm256_mul_epu32(self, scale);
            let res_even = _mm256_srli_epi64(prod_even, 32);
            let v_u32_shifted = _mm256_srli_epi64(self, 32);
            let prod_odd = _mm256_mul_epu32(v_u32_shifted, scale);
            let merged = _mm256_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
            _mm256_add_epi32(merged, min)
        }
    }

    fn to_randf(self, scale: Self::Float, min: Self::Float) -> Self::Float {
        unsafe {
            _mm256_add_ps(_mm256_mul_ps(simd_f01::u32x8(self), scale), min)
        }
    }
}

impl WordV for __m512i {
    type Int = Self;
    type Float = __m512;

    fn to_f01(self) -> Self::Float {
        unsafe { simd_f01::u32x16(self) }
    }

    fn to_randi(self, scale: Self::Int, min: Self::Int) -> Self::Int {
        const MERGE_MASK: u16 = 0b1010101010101010;

        unsafe {
            let prod_even = _mm512_mul_epu32(self, scale);
            let res_even = _mm512_srli_epi64(prod_even, 32);
            let v_u32_shifted = _mm512_srli_epi64(self, 32);
            let prod_odd = _mm512_mul_epu32(v_u32_shifted, scale);
            let merged = _mm512_mask_blend_epi32(MERGE_MASK, res_even, prod_odd);
            _mm512_add_epi32(merged, min)
        }
    }

    fn to_randf(self, scale: Self::Float, min: Self::Float) -> Self::Float {
        unsafe {
            _mm512_add_ps(_mm512_mul_ps(simd_f01::u32x16(self), scale), min)
        }
    }
}

pub trait RngV {
    type Word: WordV;

    fn nextuv(&mut self) -> Self::Word;

    fn nextfv(&mut self) -> <Self::Word as WordV>::Float {
        self.nextuv().to_f01()
    }

    fn randiv(
        &mut self,
        scale: <Self::Word as WordV>::Int,
        min: <Self::Word as WordV>::Int,
    ) -> <Self::Word as WordV>::Int {
        self.nextuv().to_randi(scale, min)
    }

    fn randfv(
        &mut self,
        scale: <Self::Word as WordV>::Float,
        min: <Self::Word as WordV>::Float,
    ) -> <Self::Word as WordV>::Float {
        self.nextuv().to_randf(scale, min)
    }
}
