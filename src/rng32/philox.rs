use crate::rng::Rng32;
use crate::rng32::SplitMix32;
use crate::wrap;
use std::num::Wrapping;

use std::arch::x86_64::*;

// --- Philox32 ---

/// A Philox 4x32 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
///
/// # Examples
///
/// ```
/// use urng::rng32::Philox32x4;
///
/// let mut rng = Philox32x4::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Philox32x4 {
    pub(crate) c: [Wrapping<u32>; 4],
    pub(crate) k: [Wrapping<u32>; 2],
}

impl Philox32x4 {
    /// Creates a new `Philox32x4` instance seeded with the given value.
    ///
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            c: wrap![
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
            ],
            k: wrap![seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline(always)]
    pub(crate) fn compute(c: [Wrapping<u32>; 4], k: [Wrapping<u32>; 2]) -> [u32; 4] {
        let mut x = [c[0].0, c[1].0, c[2].0, c[3].0];
        let mut key = wrap![k[0].0, k[1].0];

        const M0: u64 = 0xD2511F53;
        const M1: u64 = 0xCD9E8D57;
        const W0: u32 = 0x9E3779B9;
        const W1: u32 = 0xBB67AE85;

        for _ in 0..10 {
            let prod0 = (x[0] as u64).wrapping_mul(M0);
            let hi0 = (prod0 >> 32) as u32;
            let lo0 = prod0 as u32;

            let prod1 = (x[2] as u64).wrapping_mul(M1);
            let hi1 = (prod1 >> 32) as u32;
            let lo1 = prod1 as u32;

            x[0] = hi1 ^ x[1] ^ key[0].0;
            x[1] = lo1;
            x[2] = hi0 ^ x[3] ^ key[1].0;
            x[3] = lo0;

            // key[0] = key[0].wrapping_add(W0);
            // key[1] = key[1].wrapping_add(W1);
            key = [key[0] * wrap!(W0), key[1] * wrap!(W1)];
        }

        x
    }

    /// Generates the next block of random numbers.
    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; 4] {
        let out = Self::compute(self.c, self.k);
        self.c[0] += 1;
        if self.c[0].0 == 0 {
            self.c[1] += 1;
            if self.c[1].0 == 0 {
                self.c[2] += 1;
                if self.c[2].0 == 0 {
                    self.c[3] += 1;
                }
            }
        }
        out
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline(always)]
    pub fn nextf(&mut self) -> [f32; 4] {
        self.nextu()
            .map(|x| x as f32 * (1.0 / (u32::MAX as f32 + 1.0)))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline(always)]
    pub fn randi(&mut self, min: i32, max: i32) -> [i32; 4] {
        let range = (max as i64 - min as i64 + 1) as u64;
        self.nextu()
            .map(|x| ((x as u64 * range) >> 32) as i32 + min)
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline(always)]
    pub fn randf(&mut self, min: f32, max: f32) -> [f32; 4] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        self.nextu().map(|x| (x as f32 * scale) + min)
    }
}

// --- Philox32x4-10 x4 ---

#[allow(non_upper_case_globals)]
pub const PHILOX32x16: usize = 16;
#[allow(non_upper_case_globals)]
pub const PHILOX32x4x4_PAR_CHUNK: usize = 131_072;
#[allow(non_upper_case_globals)]
pub const PHILOX32x4x4_CHUNK_RATIO: u128 = (PHILOX32x4x4_PAR_CHUNK / PHILOX32x16) as u128;
#[allow(non_upper_case_globals)]
pub const PHILOX32x4x4_SHIFT: u128 = PHILOX32x4x4_CHUNK_RATIO.trailing_zeros() as u128;
#[allow(non_upper_case_globals)]
pub const PHILOX32x16_SHIFT: usize = PHILOX32x16.trailing_zeros() as usize;

#[cfg(target_arch = "x86_64")]
/// A Philox 4x32x4 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Philox32x4x4 {
    pub(crate) c: __m512i,
    pub(crate) k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Philox32x4x4 {
    /// Creates a new `Philox32x4x4` instance seeded with the given value.
    /// Requires AVX-512F support.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    ///
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut c = [0u32; PHILOX32x16];
        let mut k = [0u32; PHILOX32x16];

        let mut seedgen = SplitMix32::new(seed);
        c.iter_mut().for_each(|c| *c = seedgen.nextu());

        // [k0, 0, k1, 0]
        (0..PHILOX32x16).step_by(4).for_each(|i| {
            k[i + 0] = seedgen.nextu();
            k[i + 1] = 0;
            k[i + 2] = seedgen.nextu();
            k[i + 3] = 0;
        });

        unsafe {
            Self {
                c: _mm512_loadu_si512(c.as_ptr() as *const _),
                k: _mm512_loadu_si512(k.as_ptr() as *const _),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(crate) fn compute(&mut self) -> [u32; PHILOX32x16] {
        let mut x = self.c;
        let mut key = self.k;
        let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
        let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);

        for _ in 0..10 {
            // x0 * M0, x2 * M1 = [lo0, hi0, lo1, hi1]
            let prod = _mm512_mul_epu32(x, m);

            // shuffle -> [hi1, lo1, hi0, lo0]
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);

            // x >> 32 -> [x1, 0, x3, 0]
            let x_shift = _mm512_srli_epi64(x, 32);

            // x ^ x_shift ^ key
            x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, key));

            // key += w
            key = _mm512_add_epi32(key, w);
        }

        unsafe {
            let mut out = [0u32; PHILOX32x16];
            _mm512_storeu_si512(out.as_mut_ptr() as *mut _, x);
            out
        }
    }

    /// Generates the next block of random numbers.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; PHILOX32x16] {
        let out = self.compute();

        // increment counter
        // [1, 1, 1, 1, 1, 1, 1, 1]
        let one = _mm512_set1_epi64(1);

        // lower 64 bits (indices 0, 2, 4, 6) +1
        let next_c = _mm512_mask_add_epi64(self.c, 0x55, self.c, one);

        // check overflow of lower 64 bits
        let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
        let carry_mask = (eq_zero_mask & 0x55) << 1;

        if carry_mask != 0 {
            // if overflow, add to upper 64 bits
            self.c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        } else {
            self.c = next_c;
        }

        out
    }

    /// Generates 16 random `f32` values in the range [0, 1) using AVX-512.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf(&mut self) -> [f32; PHILOX32x16] {
        /*
        let out = self.nextu();
        let mut dst = [0f32; PHILOX32x16];
        let scale = 1.0 / (u32::MAX as f32 + 1.0);
        for i in 0..PHILOX32x16 {
            dst[i] = (out[i] as f32) * scale;
        }
        dst
        */

        unsafe {
            let out = self.nextu();
            let v_u32 = _mm512_loadu_si512(out.as_ptr() as *const _);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
            let v_res = _mm512_mul_ps(v_f32, scale);

            let mut res = [0f32; PHILOX32x16];
            _mm512_storeu_ps(res.as_mut_ptr(), v_res);
            res
        }
    }

    /// Generates a random `i32` value in the range [min, max].
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi(&mut self, min: i32, max: i32) -> [i32; PHILOX32x16] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = unsafe { self.nextu() };
        let mut dst = [0i32; PHILOX32x16];
        for i in 0..PHILOX32x16 {
            dst[i] = ((out[i] as u64 * range) >> 32) as i32 + min;
        }
        dst
    }

    /// Generates a random `f32` value in the range [min, max).
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf(&mut self, min: f32, max: f32) -> [f32; PHILOX32x16] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = unsafe { self.nextu() };
        let mut dst = [0f32; PHILOX32x16];
        for i in 0..PHILOX32x16 {
            dst[i] = (out[i] as f32 * scale) + min;
        }
        dst
    }
}

// -- Philox32 --

/// Opaque handle for the Philox32 RNG.
/// Dispatched at runtime to AVX-512 (`Philox32x4x4`) or scalar (`Philox32x4`) implementation.
///
/// # Examples
///
/// ```
/// use urng::rng32::Philox32;
///
/// let _ = core::mem::size_of::<Philox32>();
/// ```
#[repr(C)]
pub struct Philox32([u8; 0]);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Philox32x4);
    crate::unsafe_test!(Philox32x4x4);
}
