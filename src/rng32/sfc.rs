use std::arch::x86_64::*;

use wrapn::Wrap;

use crate::{_internal::FSCALE32, rng::Rng32, rng32::SplitMix32};

/// A SFC32 pseudo-random number generator.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Sfc32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc32 {
    pub a: Wrap<u32>,
    pub b: Wrap<u32>,
    pub c: Wrap<u32>,
    pub counter: Wrap<u32>,
}

impl Sfc32 {
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            a: seedgen.nextu().into(),
            b: seedgen.nextu().into(),
            c: seedgen.nextu().into(),
            counter: 1.into(),
        }
    }
}

impl Rng32 for Sfc32 {
    #[inline(always)]
    fn nextu(&mut self) -> u32 {
        let tmp = self.a + self.b + self.counter;
        self.counter += 1;
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c + (self.c << 3);
        self.c = self.c.rotate_right(11);
        self.c += tmp;
        tmp.value()
    }
}

pub(crate) const SFC32X4: usize = 4;

/// A SFC32 pseudo-random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng32::sfc::Sfc32x4;
///
/// let mut rng = Sfc32x4::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc32x4 {
    pub(crate) a: __m128i,
    pub(crate) b: __m128i,
    pub(crate) c: __m128i,
    pub(crate) counter: __m128i,
}

impl Sfc32x4 {
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut a = [0u32; SFC32X4];
        let mut b = [0u32; SFC32X4];
        let mut c = [0u32; SFC32X4];
        for i in 0..SFC32X4 {
            a[i] = seedgen.nextu();
            b[i] = seedgen.nextu();
            c[i] = seedgen.nextu();
        }

        unsafe {
            Self {
                a: _mm_loadu_si128(a.as_ptr() as *const _),
                b: _mm_loadu_si128(b.as_ptr() as *const _),
                c: _mm_loadu_si128(c.as_ptr() as *const _),
                counter: _mm_set1_epi32(1),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn nextuv(&mut self) -> __m128i {
        unsafe {
            let tmp = _mm_add_epi32(_mm_add_epi32(self.a, self.b), self.counter);
            self.counter = _mm_add_epi32(self.counter, _mm_set1_epi32(1));
            self.a = _mm_xor_si128(self.b, _mm_srli_epi32(self.b, 9));
            self.b = _mm_add_epi32(self.c, _mm_slli_epi32(self.c, 3));
            self.c = _mm_add_epi32(
                _mm_or_si128(_mm_slli_epi32(self.c, 21), _mm_srli_epi32(self.c, 11)),
                tmp,
            );
            tmp
        }
    }

    #[inline(always)]
    fn u32x4_to_ps(v: __m128i) -> __m128 {
        unsafe {
            let lo = _mm_and_si128(v, _mm_set1_epi32(0xffff));
            let hi = _mm_srli_epi32(v, 16);
            _mm_add_ps(
                _mm_mul_ps(_mm_cvtepi32_ps(hi), _mm_set1_ps(65536.0)),
                _mm_cvtepi32_ps(lo),
            )
        }
    }

    #[inline(always)]
    pub(crate) fn nextfv(&mut self, scale: __m128) -> __m128 {
        unsafe { _mm_mul_ps(Self::u32x4_to_ps(self.nextuv()), scale) }
    }

    #[inline(always)]
    pub(crate) fn randiv(&mut self, v_range: __m128i, v_min: __m128i) -> __m128i {
        unsafe {
            let uv = self.nextuv();
            let hi = _mm_set1_epi64x((0xffff_ffffu64 << 32) as i64);
            let res_even = _mm_srli_epi64(_mm_mul_epu32(uv, v_range), 32);
            let prod_odd = _mm_and_si128(
                _mm_mul_epu32(_mm_srli_epi64(uv, 32), _mm_srli_epi64(v_range, 32)),
                hi,
            );
            _mm_add_epi32(_mm_or_si128(res_even, prod_odd), v_min)
        }
    }

    #[inline(always)]
    pub(crate) fn randfv(&mut self, v_mult: __m128, v_min: __m128) -> __m128 {
        unsafe { _mm_add_ps(_mm_mul_ps(Self::u32x4_to_ps(self.nextuv()), v_mult), v_min) }
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; SFC32X4] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    #[inline(always)]
    pub fn nextf(&mut self) -> [f32; SFC32X4] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

pub(crate) const SFC32X8: usize = 8;

/// A SFC32 pseudo-random number generator.
///
/// # Examples
///
/// ```no_run
/// use urng::rng32::sfc::Sfc32x8;
///
/// let mut rng = unsafe { Sfc32x8::new(1) };
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc32x8 {
    pub(crate) a: __m256i,
    pub(crate) b: __m256i,
    pub(crate) c: __m256i,
    pub(crate) counter: __m256i,
}

#[allow(dead_code)]
impl Sfc32x8 {
    ///
    /// # Safety
    ///
    /// The caller must ensure the CPU supports the `avx2` target feature.
    #[target_feature(enable = "avx2")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut a = [0u32; SFC32X8];
        let mut b = [0u32; SFC32X8];
        let mut c = [0u32; SFC32X8];
        for i in 0..SFC32X8 {
            a[i] = seedgen.nextu();
            b[i] = seedgen.nextu();
            c[i] = seedgen.nextu();
        }
        unsafe {
            Self {
                a: _mm256_loadu_si256(a.as_ptr() as *const _),
                b: _mm256_loadu_si256(b.as_ptr() as *const _),
                c: _mm256_loadu_si256(c.as_ptr() as *const _),
                counter: _mm256_set1_epi32(1),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn nextuv(&mut self) -> __m256i {
        let tmp = _mm256_add_epi32(_mm256_add_epi32(self.a, self.b), self.counter);
        self.counter = _mm256_add_epi32(self.counter, _mm256_set1_epi32(1));
        self.a = _mm256_xor_si256(self.b, _mm256_srli_epi32(self.b, 9));
        self.b = _mm256_add_epi32(self.c, _mm256_slli_epi32(self.c, 3));
        self.c = _mm256_add_epi32(unsafe { _mm256_rol_epi32(self.c, 21) }, tmp);
        tmp
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn nextfv(&mut self, scale: __m256) -> __m256 {
        let v_f32 = _mm256_cvtepi32_ps(unsafe { self.nextuv() });
        _mm256_mul_ps(v_f32, scale)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn randiv(&mut self, v_range: __m256i, v_min: __m256i) -> __m256i {
        const MERGE_MASK: u8 = 0xAA;
        let v_u32 = unsafe { self.nextuv() };
        let res_even = _mm256_srli_epi64(_mm256_mul_epu32(v_u32, v_range), 32);
        let prod_odd = _mm256_mul_epu32(_mm256_srli_epi64(v_u32, 32), v_range);
        let merged = unsafe { _mm256_mask_blend_epi32(MERGE_MASK, res_even, prod_odd) };
        _mm256_add_epi32(merged, v_min)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn randfv(&mut self, v_mult: __m256, v_min: __m256) -> __m256 {
        let v_f32 = _mm256_cvtepi32_ps(unsafe { self.nextuv() });
        _mm256_add_ps(_mm256_mul_ps(v_f32, v_mult), v_min)
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; SFC32X8] {
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    #[inline(always)]
    pub fn nextf(&mut self) -> [f32; SFC32X8] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

pub(crate) const SFC32X16: usize = 16;

/// A SFC32 pseudo-random number generator.
///
/// # Examples
///
/// ```no_run
/// use urng::rng32::sfc::Sfc32x16;
///
/// let mut rng = unsafe { Sfc32x16::new(1) };
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc32x16 {
    pub(crate) a: __m512i,
    pub(crate) b: __m512i,
    pub(crate) c: __m512i,
    pub(crate) counter: __m512i,
}

#[allow(dead_code)]
impl Sfc32x16 {
    ///
    /// # Safety
    ///
    /// The caller must ensure the CPU supports the `avx512f` target feature.
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
    pub(crate) unsafe fn nextfv(&mut self, scale: __m512) -> __m512 {
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
        unsafe { std::mem::transmute(self.nextuv()) }
    }

    pub fn nextf(&mut self) -> [f32; SFC32X16] {
        self.nextu().map(|x| x as f32 * FSCALE32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe_test;
    #[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
    use crate::unsafe_test;

    safe_test!(Sfc32);
    safe_test!(Sfc32x4);
    #[cfg(target_feature = "avx2")]
    unsafe_test!(Sfc32x8);
    #[cfg(target_feature = "avx512f")]
    unsafe_test!(Sfc32x16);

    fn scalar_lanes(seed: u32) -> [Sfc32; SFC32X4] {
        let mut seedgen = SplitMix32::new(seed);
        std::array::from_fn(|_| Sfc32 {
            a: seedgen.nextu().into(),
            b: seedgen.nextu().into(),
            c: seedgen.nextu().into(),
            counter: 1.into(),
        })
    }

    #[test]
    fn sfc32x4_lanes_match_scalar() {
        let mut vector = Sfc32x4::new(0);
        let mut scalars = scalar_lanes(0);

        for _ in 0..8 {
            let got = vector.nextu();
            let want: [u32; SFC32X4] = std::array::from_fn(|i| scalars[i].nextu());
            assert_eq!(got, want);
        }
    }

    #[test]
    fn sfc32x4_randiv_randfv_match_scalar() {
        let (min_i, max_i) = (-5i32, 7i32);
        let (min_f, max_f) = (-2.0f32, 3.0f32);
        let v_range = unsafe { _mm_set1_epi32((max_i as i64 - min_i as i64 + 1) as u32 as i32) };
        let v_min = unsafe { _mm_set1_epi32(min_i) };
        let v_mult = unsafe { _mm_set1_ps((max_f - min_f) * FSCALE32) };
        let v_minf = unsafe { _mm_set1_ps(min_f) };
        let scale = unsafe { _mm_set1_ps(FSCALE32) };

        let mut vector = Sfc32x4::new(1);
        let mut scalars = scalar_lanes(1);
        for _ in 0..8 {
            let got: [i32; SFC32X4] = unsafe { std::mem::transmute(vector.randiv(v_range, v_min)) };
            let want: [i32; SFC32X4] = std::array::from_fn(|i| scalars[i].randi(min_i, max_i));
            assert_eq!(got, want);
        }

        let mut vector = Sfc32x4::new(2);
        let mut scalars = scalar_lanes(2);
        for _ in 0..8 {
            let got: [f32; SFC32X4] = unsafe { std::mem::transmute(vector.randfv(v_mult, v_minf)) };
            let want: [f32; SFC32X4] = std::array::from_fn(|i| scalars[i].randf(min_f, max_f));
            assert_eq!(got, want);
        }

        let mut vector = Sfc32x4::new(3);
        let mut scalars = scalar_lanes(3);
        for _ in 0..8 {
            let got: [f32; SFC32X4] = unsafe { std::mem::transmute(vector.nextfv(scale)) };
            let want: [f32; SFC32X4] = std::array::from_fn(|i| scalars[i].nextf());
            assert_eq!(got, want);
        }
    }
}
