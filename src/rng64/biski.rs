use crate::{rng::Rng64, rng64::SplitMix64};
use std::arch::x86_64::*;

/// A [Biski64](https://github.com/danielcota/biski64) random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng::Rng64;
/// use urng::rng64::Biski64;
///
/// let mut rng = Biski64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Biski64 {
    fast_loop: u64,
    mix: u64,
    loop_mix: u64,
}

impl Biski64 {
    /// Creates a new `Biski64` instance seeded via `SplitMix64`.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            fast_loop: seedgen.nextu(),
            mix: seedgen.nextu(),
            loop_mix: seedgen.nextu(),
        }
    }
}

impl Rng64 for Biski64 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        let output = self.mix.wrapping_add(self.loop_mix);

        (self.fast_loop, self.mix, self.loop_mix) = (
            self.fast_loop.wrapping_add(0x9999999999999999),
            self.mix
                .rotate_left(16)
                .wrapping_add(self.loop_mix.rotate_left(40)),
            self.fast_loop ^ self.mix,
        );

        output
    }
}

/// A 4-way SIMD Biski64 generator using AVX512 512-bit intrinsics.
///
/// # Examples
///
/// ```
/// use urng::rng64::Biski64x8;
///
/// unsafe {
///     let mut rng = Biski64x8::new(0);
///     let _ = rng.nextu();
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Biski64x8 {
    fast_loop: __m512i,
    mix: __m512i,
    loop_mix: __m512i,
}

pub(crate) const INC: u64 = 0x9999999999999999;

#[cfg(target_arch = "x86_64")]
impl Biski64x8 {
    /// Creates a new `Biski64x8` from 8 independent seeds.
    ///
    /// # Safety
    /// Requires AVX512 support (guaranteed by `target-cpu=native` on modern x86_64).
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        let mut fast_loop = [0u64; 8];
        let mut mix = [0u64; 8];
        let mut loop_mix = [0u64; 8];
        let mut sg = SplitMix64::new(seed);
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
    #[inline(always)]
    pub unsafe fn nextf(&mut self) -> [f64; 8] {
        unsafe {
            let u = self.nextu();
            let mut out = [0f64; 8];
            let scale = 1.0 / (u64::MAX as f64 + 1.0);
            for i in 0..8 {
                out[i] = u[i] as f64 * scale;
            }
            out
        }
    }

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
    #[inline(always)]
    pub unsafe fn randf(&mut self, min: f64, max: f64) -> [f64; 8] {
        unsafe {
            let u = self.nextu();
            let range = max - min;
            let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
            let mut out = [0f64; 8];
            for i in 0..8 {
                out[i] = (u[i] as f64 * scale) + min;
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Biski64);
    crate::unsafe_test!(Biski64x8);
}
