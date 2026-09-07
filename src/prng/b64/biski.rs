#[cfg(feature = "simd")]
use std::arch::x86_64::*;

use wrapn::{wrap, wu64};

#[cfg(feature = "simd")]
use crate::_internal::{i2f_bits, u2f_01};
use crate::{prng::b64::SplitMix64, rng::Rng};

/// Biski64 64-bit RNG implementation.
///
/// # Example
/// ```
/// use urng::{Rng, Biski64};
///
/// let mut rng = Biski64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Biski64 {
    fast_loop: wu64,
    mix: wu64,
    loop_mix: wu64,
}

impl Biski64 {
    /// Creates a new `Biski64` instance with the given seed.
    pub const fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            fast_loop: wrap!(seedgen.nextu_const()),
            mix: wrap!(seedgen.nextu_const()),
            loop_mix: wrap!(seedgen.nextu_const()),
        }
    }
}

impl Rng for Biski64 {
    type Word = u64;

    #[inline(always)]
    fn nextu(&mut self) -> Self::Word {
        let output = self.mix + self.loop_mix;

        (self.fast_loop, self.mix, self.loop_mix) = (
            self.fast_loop + 0x9999999999999999,
            self.mix.rotate_left(16) + self.loop_mix.rotate_left(40),
            self.fast_loop ^ self.mix,
        );

        output.value()
    }
}

/// A 4-way SIMD Biski64 generator using AVX512 512-bit intrinsics.
///
/// # Example
/// ```no_run
/// use urng::Biski64x8;
///
/// let mut rng = Biski64x8::new(0);
/// let _ = unsafe { rng.nextu() };
/// ```
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[repr(C, align(64))]
pub struct Biski64x8 {
    fast_loop: __m512i,
    mix: __m512i,
    loop_mix: __m512i,
}

#[cfg(feature = "simd")]
pub(crate) const INC: u64 = 0x9999999999999999;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Biski64x8 {
    /// Creates a new `Biski64x8` from 8 independent seeds.
    ///
    /// # Safety
    /// Requires AVX512 support (guaranteed by `target-cpu=native` on modern x86_64).
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        let mut sg = SplitMix64::new(seed);
        let mut fast_loop = [0u64; 8];
        let mut mix = [0u64; 8];
        let mut loop_mix = [0u64; 8];
        for i in 0..8 {
            fast_loop[i] = sg.nextu();
            mix[i] = sg.nextu();
            loop_mix[i] = sg.nextu();
        }
        unsafe {
            Self {
                fast_loop: _mm512_loadu_si512(fast_loop.as_ptr() as *const __m512i),
                mix: _mm512_loadu_si512(mix.as_ptr() as *const __m512i),
                loop_mix: _mm512_loadu_si512(loop_mix.as_ptr() as *const __m512i),
            }
        }
    }

    /// Generates 4 random `u64` values simultaneously and writes them to `out`.
    ///
    /// # Safety
    /// `out` must point to a valid buffer of at least 8 `u64` values.
    /// Requires AVX512 support.
    #[inline(always)]
    pub unsafe fn nextu(&mut self) -> [u64; 8] {
        unsafe {
            // let output = self.mix.wrapping_add(self.loop_mix);
            let output = _mm512_add_epi64(self.mix, self.loop_mix);

            let inc = _mm512_set1_epi64(INC as i64);
            let fast_loop = _mm512_add_epi64(self.fast_loop, inc);
            let mix = _mm512_add_epi64(
                _mm512_rol_epi64(self.mix, 16),
                _mm512_rol_epi64(self.loop_mix, 40),
            );
            self.fast_loop = fast_loop;
            self.mix = mix;
            self.loop_mix = _mm512_xor_si512(self.fast_loop, self.mix);

            let mut res = [0u64; 8];
            _mm512_storeu_si512(res.as_mut_ptr() as *mut __m512i, output);
            res
        }
    }

    /// Generates 8 random `f64` values in [0, 1) and writes them to `out`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the CPU supports the `avx512f` target feature.
    #[inline(always)]
    pub unsafe fn nextf(&mut self) -> [f64; 8] {
        unsafe {
            let u = self.nextu();
            let mut out = [0f64; 8];
            for i in 0..8 {
                out[i] = u2f_01!(f64, 64, u[i]);
            }
            out
        }
    }

    /// Generates 8 random `i64` values in [min, max].
    ///
    /// # Safety
    ///
    /// The caller must ensure the CPU supports the `avx512f` target feature.
    #[inline(always)]
    pub unsafe fn randi(&mut self, min: i64, max: i64) -> [i64; 8] {
        unsafe {
            let u = self.nextu();
            let range = (max as i128 - min as i128 + 1) as u128;
            let mut out = [0i64; 8];
            for i in 0..8 {
                out[i] = ((u[i] as u128 * range) >> 64) as i64 + min;
            }
            out
        }
    }

    /// Generates 8 random `f64` values in [min, max) and writes them to `out`.
    ///
    /// # Safety
    ///
    /// The caller must ensure the CPU supports the `avx512f` target feature.
    #[inline(always)]
    pub unsafe fn randf(&mut self, min: f64, max: f64) -> [f64; 8] {
        unsafe {
            let u = self.nextu();
            let range = max - min;
            let mut out = [0f64; 8];
            for i in 0..8 {
                out[i] = u2f_01!(f64, 64, u[i]) * range + min;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Biski64);

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    crate::unsafe_test!(Biski64x8);
}
