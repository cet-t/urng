use crate::{_internal::FSCALE32, rng::Rng32, rng32::SplitMix32};
use std::arch::x86_64::*;
use wide::{f32x4, i32x4, u32x4, u64x2};

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
    pub(crate) a: u32x4,
    pub(crate) b: u32x4,
    pub(crate) c: u32x4,
    pub(crate) counter: u32x4,
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

        Self {
            a: u32x4::from(a),
            b: u32x4::from(b),
            c: u32x4::from(c),
            counter: u32x4::splat(1),
        }
    }

    #[inline(always)]
    pub(crate) fn nextuv(&mut self) -> u32x4 {
        let tmp = self.a + self.b + self.counter;
        self.counter += u32x4::splat(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c + (self.c << 3);
        self.c = ((self.c << 21) | (self.c >> (32 - 21))) + tmp;
        tmp
    }

    #[inline(always)]
    pub(crate) fn nextfv(&mut self, scale: f32x4) -> f32x4 {
        let arr: [u32; 4] = bytemuck::cast(self.nextuv());
        f32x4::from([arr[0] as f32, arr[1] as f32, arr[2] as f32, arr[3] as f32]) * scale
    }

    #[inline(always)]
    pub(crate) fn randiv(&mut self, v_range: u32x4, v_min: i32x4) -> i32x4 {
        let v: u64x2 = bytemuck::cast(self.nextuv());
        let r: u64x2 = bytemuck::cast(v_range);
        let lo = u64x2::splat(0xffff_ffff);
        let res_even: u32x4 = bytemuck::cast((v & lo) * (r & lo) >> 32);
        let prod_odd: u32x4 = bytemuck::cast((v >> 32) * (r >> 32) & (lo << 32));
        let merged: i32x4 = bytemuck::cast(res_even | prod_odd);
        merged + v_min
    }

    #[inline(always)]
    pub(crate) fn randfv(&mut self, v_mult: f32x4, v_min: f32x4) -> f32x4 {
        let v_f32 = f32x4::from_i32x4(bytemuck::cast(self.nextuv()));
        v_f32 * v_mult + v_min
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; SFC32X4] {
        bytemuck::cast(self.nextuv())
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
/// ```
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

impl Sfc32x8 {
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
/// ```
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
    use crate::{safe_test, unsafe_test};

    safe_test!(Sfc32);
    safe_test!(Sfc32x4);
    unsafe_test!(Sfc32x8);
    unsafe_test!(Sfc32x16);
}
