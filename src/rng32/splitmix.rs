use crate::{rng::Rng32, wrap};
use std::num::Wrapping;

use std::arch::x86_64::*;

/// A SplitMix32 pseudo-random number generator.
///
/// Fast 32-bit finalizer-based PRNG commonly used to seed other generators.
/// Uses a single 32-bit state word advanced by the golden-ratio constant.
///
/// # Examples
///
/// ```
/// use urng::rng32::SplitMix32;
///
/// let mut rng = SplitMix32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct SplitMix32 {
    state: Wrapping<u32>,
}

const A: Wrapping<u64> = wrap!(0xFF51_AFD7_ED55_8CCD);
const B: Wrapping<u64> = wrap!(0xC4CE_B9FE_1A85_EC53);

impl SplitMix32 {
    /// Creates a new `SplitMix32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// assert_ne!(rng.nextu(), 0);
    /// ```
    pub fn new(seed: u32) -> Self {
        Self {
            state: wrap!(seed | 1),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// assert_ne!(rng.nextu(), 0);
    /// ```
    pub fn nextu(&mut self) -> u32 {
        self.state += wrap!(0x9E3779B9);

        let mut z = wrap!(self.state.0 as u64);
        z = (z ^ (z >> 16)) * A;
        z = (z ^ (z >> 16)) * B;
        (z ^ (z >> 16)).0 as u32
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// let val: f32 = rng.randf(0.0, 1.0);
    /// assert!(val >= 0.0 && val < 1.0);
    /// ```
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for SplitMix32 {
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }

    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }

    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[allow(non_upper_case_globals)]
pub const SPLITMIX32x16: usize = 16;
#[allow(non_upper_case_globals)]
pub const SPLITMIX32x16_PAR_CHUNK: usize = 8192;
pub const SPLITMIX32_GAMMA: u32 = 0x9E37_79B9;

/// AVX-512 implementation of SplitMix32 producing 16 values per step.
///
/// # Examples
///
/// ```ignore
/// use urng::rng32::SplitMix32x16;
///
/// let mut rng = unsafe { SplitMix32x16::new(1) };
/// let vals = unsafe { rng.nextu() };
/// assert_eq!(vals.len(), 16);
/// ```
/// AVX-512 implementation of SplitMix32 producing 16 values per step.
///
/// # Examples
///
/// ```ignore
/// use urng::rng32::SplitMix32x16;
///
/// let mut rng = unsafe { SplitMix32x16::new(1) };
/// let vals = unsafe { rng.nextu() };
/// assert_eq!(vals.len(), 16);
/// ```
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct SplitMix32x16 {
    pub(crate) state: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl SplitMix32x16 {
    /// Creates a new `SplitMix32x16` instance.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let base = seed | 1;
        let mut init = [0u32; SPLITMIX32x16];
        for (i, v) in init.iter_mut().enumerate() {
            *v = base.wrapping_add(SPLITMIX32_GAMMA.wrapping_mul((i as u32).wrapping_add(1)));
        }
        Self {
            state: unsafe { _mm512_loadu_si512(init.as_ptr() as *const _) },
        }
    }

    #[target_feature(enable = "avx512f")]
    /// Computes the SplitMix32 output for 16 lanes at once.
    pub unsafe fn compute(state: __m512i) -> __m512i {
        let c1 = _mm512_set1_epi32(0x85EB_CA6Bu32 as i32);
        let c2 = _mm512_set1_epi32(0xC2B2_AE35u32 as i32);

        let mut z = state;
        z = _mm512_xor_si512(z, _mm512_srli_epi32(z, 16));
        z = _mm512_add_epi32(z, c1);
        z = _mm512_xor_si512(z, _mm512_srli_epi32(z, 13));
        z = _mm512_add_epi32(z, c2);
        _mm512_xor_si512(z, _mm512_srli_epi32(z, 16))
    }

    #[target_feature(enable = "avx512f")]
    /// Generates the next 16 random `u32` values.
    pub unsafe fn nextu(&mut self) -> [u32; SPLITMIX32x16] {
        let v = unsafe { Self::compute(self.state) };
        self.state = _mm512_add_epi32(
            self.state,
            _mm512_set1_epi32(SPLITMIX32_GAMMA.wrapping_mul(SPLITMIX32x16 as u32) as i32),
        );
        let mut out = [0u32; SPLITMIX32x16];
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut _, v) };
        out
    }
}

// -- SplitMix32Simd --

/// Opaque handle for the SplitMix32 RNG.
/// Dispatched at runtime to AVX-512 (`SplitMix32x16`) or scalar (`SplitMix32`) implementation.
///
/// # Examples
///
/// ```
/// use urng::rng32::SplitMix32Simd;
///
/// let _ = core::mem::size_of::<SplitMix32Simd>();
/// ```
#[repr(C)]
pub struct SplitMix32Simd([u8; 0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix32_works() {
        let mut rng = SplitMix32::new(1);
        assert_eq!(rng.nextu(), 2338703698);
        assert_eq!(rng.nextf(), 0.80941534);
    }
}
