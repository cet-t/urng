use crate::rng64::SplitMix64;

use std::arch::x86_64::*;

// --- Squares32 ---

/// The Squares random number generator (32-bit output version by Bernard Widynski).
pub struct Squares32 {
    pub(crate) c: u64,
    pub(crate) k: u64,
}

impl Squares32 {
    /// Creates a new `Squares32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Squares32;
    ///
    /// let mut rng = Squares32::new(1);
    /// assert_eq!(rng.nextu(), 1225738608);
    /// ```
    #[inline]
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Self {
            c: 0,
            k: seedgen.nextu(),
        }
    }

    /// Core computation: 4 rounds of middle-square with counter.
    /// Takes pre-computed y = ctr*key and z = y + key to avoid
    /// redundant multiplication in batch scenarios.
    #[inline(always)]
    pub fn compute_yz(y: u64, z: u64) -> u32 {
        let mut x: u64;

        x = y.wrapping_mul(y).wrapping_add(y);
        x = x.rotate_left(32);

        x = x.wrapping_mul(x).wrapping_add(z);
        x = x.rotate_left(32);

        x = x.wrapping_mul(x).wrapping_add(y);
        x = x.rotate_left(32);

        (x.wrapping_mul(x).wrapping_add(z) >> 32) as u32
    }

    /// Convenience wrapper: compute from counter and key directly.
    #[inline(always)]
    pub fn compute(ctr: u64, key: u64) -> u32 {
        let y = ctr.wrapping_mul(key);
        let z = y.wrapping_add(key);
        Self::compute_yz(y, z)
    }

    /// Generates the next random `u32` value.
    #[inline(always)]
    pub fn nextu(&mut self) -> u32 {
        let out = Self::compute(self.c, self.k);
        self.c = self.c.wrapping_add(1);
        out
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline(always)]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline(always)]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline(always)]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }
}

// C-ABI exports for Squares32

#[allow(non_upper_case_globals)]
pub const SQUARES32x8: usize = 8;

/// A high-throughput Squares random number generator utilizing AVX-512 SIMD instructions.
/// This implementation processes 8 counters in parallel and is highly optimized with 4-way unrolling.
#[cfg(target_arch = "x86_64")]
#[repr(C)]
#[repr(align(64))]
pub struct Squares32x8 {
    /// 8 counters stored in a 512-bit SIMD register.
    pub c: __m512i,
    /// 8 keys stored in a 512-bit SIMD register.
    pub k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Squares32x8 {
    /// Creates a new `Squares32x8` instance from a 32-bit seed.
    /// The seed is used to initialize the counters and keys.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut k = [0u64; SQUARES32x8];
        let mut seedgen = SplitMix64::new(seed as u64 | 1);
        k.iter_mut().for_each(|v| {
            *v = seedgen.nextu();
        });

        unsafe {
            Self {
                c: _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
                k: _mm512_loadu_si512(k.as_ptr() as *const _),
            }
        }
    }

    /// Core computation: 4 rounds of middle-square with counter.
    /// Returns 8x u32 random values in the lower 32-bits of each 64-bit lane.
    ///
    /// # Arguments
    /// * `y` - Pre-computed y = ctr * key.
    /// * `z` - Pre-computed z = y + key.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn compute_yz(y: __m512i, z: __m512i) -> __m256i {
        let mut x = _mm512_add_epi64(_mm512_mullo_epi64(y, y), y);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), z);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), y);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        _mm512_cvtepi64_epi32(_mm512_srli_epi64(
            _mm512_add_epi64(_mm512_mullo_epi64(x, x), z),
            32,
        ))
    }

    /// Convenience wrapper to compute random values from counter and key directly.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) unsafe fn compute(c: __m512i, k: __m512i) -> __m256i {
        unsafe {
            let y = _mm512_mullo_epi64(c, k);
            let z = _mm512_add_epi64(y, k);
            Self::compute_yz(y, z)
        }
    }

    /// Generates 8 new `u32` random numbers.
    /// Increments the internal counters by 8.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn nextu(&mut self) -> [u32; SQUARES32x8] {
        unsafe {
            let v = Self::compute(self.c, self.k);
            self.c = _mm512_add_epi64(self.c, _mm512_set1_epi64(8));
            let mut out = [0u32; SQUARES32x8];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut _, v);
            out
        }
    }
}

// -- Squares32Simd --

/// Opaque handle for the Squares32 RNG.
/// Dispatched at runtime to AVX-512 (`Squares32x8`) or scalar (`Squares32`) implementation.
#[repr(C)]
pub struct Squares32Simd([u8; 0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squares32_works() {
        let mut rng = Squares32::new(1);
        assert_eq!(rng.nextu(), 1225738608);
        assert_eq!(rng.nextf(), 0.9183048);
    }

    #[test]
    fn squares32x8_works() {
        unsafe {
            let mut rng = Squares32x8::new(1);
            assert_eq!(
                rng.nextu(),
                [
                    1225738608, 3081786017, 2002165410, 1518623550, 443612158, 1744152856,
                    1924491776, 1460635941
                ]
            );
        }
    }
}
