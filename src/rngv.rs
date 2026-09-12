use std::arch::x86_64::*;

mod sealed {
    use std::arch::x86_64::{__m128i, __m256i, __m512i};

    pub trait Sealed {}
    impl Sealed for __m128i {}
    impl Sealed for __m256i {}
    impl Sealed for __m512i {}
}

pub trait VWord: sealed::Sealed + Copy + Sized {
    type Int;
    type Float;

    #[must_use]
    fn to_f01(self, scale: Self::Float) -> Self::Float;

    #[must_use]
    fn to_randi(self, min: Self::Int, scale: Self::Int) -> Self::Int;

    #[must_use]
    fn to_randf(self, min: Self::Float, scale: Self::Float) -> Self::Float;
}

impl VWord for __m128i {
    type Int = Self;
    type Float = __m128;

    fn to_f01(self, scale: Self::Float) -> Self::Float {
        let f = unsafe { _mm_cvtepi32_ps(self) };
        unsafe { _mm_mul_ps(f, scale) }
    }

    fn to_randi(self, min: Self::Int, scale: Self::Int) -> Self::Int {
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

    fn to_randf(self, min: Self::Float, scale: Self::Float) -> Self::Float {
        unsafe {
            let fv = _mm_cvtepi32_ps(self);
            _mm_add_ps(_mm_mul_ps(fv, scale), min)
        }
    }
}

impl VWord for __m256i {
    type Int = Self;
    type Float = __m256;

    fn to_f01(self, scale: Self::Float) -> Self::Float {
        let f = unsafe { _mm256_cvtepi32_ps(self) };
        unsafe { _mm256_mul_ps(f, scale) }
    }

    fn to_randi(self, min: Self::Int, scale: Self::Int) -> Self::Int {
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

    fn to_randf(self, min: Self::Float, scale: Self::Float) -> Self::Float {
        unsafe {
            let fv = _mm256_cvtepi32_ps(self);
            _mm256_add_ps(_mm256_mul_ps(fv, scale), min)
        }
    }
}

impl VWord for __m512i {
    type Int = Self;
    type Float = __m512;

    fn to_f01(self, scale: Self::Float) -> Self::Float {
        unsafe {
            let f = _mm512_cvtepu32_ps(self);
            _mm512_mul_ps(f, scale)
        }
    }

    fn to_randi(self, min: Self::Int, scale: Self::Int) -> Self::Int {
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

    fn to_randf(self, min: Self::Float, scale: Self::Float) -> Self::Float {
        unsafe {
            let fv = _mm512_cvtepu32_ps(self);
            _mm512_add_ps(_mm512_mul_ps(fv, scale), min)
        }
    }
}

pub trait VRng {
    type Word: VWord;

    fn nextuv(&mut self) -> Self::Word;

    fn nextfv(&mut self, scale: <Self::Word as VWord>::Float) -> <Self::Word as VWord>::Float {
        self.nextuv().to_f01(scale)
    }

    fn randiv(
        &mut self,
        min: <Self::Word as VWord>::Int,
        max: <Self::Word as VWord>::Int,
    ) -> <Self::Word as VWord>::Int {
        self.nextuv().to_randi(min, max)
    }

    fn randfv(
        &mut self,
        min: <Self::Word as VWord>::Float,
        max: <Self::Word as VWord>::Float,
    ) -> <Self::Word as VWord>::Float {
        self.nextuv().to_randf(min, max)
    }
}
