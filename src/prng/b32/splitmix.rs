use wrapn::{wrap, wu32};

use crate::rng::Rng;

/// SplitMix32 32-bit RNG implementation.
///
/// Fast 32-bit finalizer-based PRNG commonly used to seed other generators.
/// Uses a single 32-bit state word advanced by the golden-ratio constant.
///
/// # Example
/// ```
/// use urng::{Rng, SplitMix32};
///
/// let mut rng = SplitMix32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SplitMix32 {
    state: wu32,
}

const A: u64 = 0xFF51_AFD7_ED55_8CCD;
const B: u64 = 0xC4CE_B9FE_1A85_EC53;

impl SplitMix32 {
    /// Creates a new `SplitMix32` instance with the given seed.
    pub const fn new(seed: u32) -> Self {
        Self {
            state: wrap!(seed | 1),
        }
    }

    #[inline(always)]
    pub(crate) const fn nextu_const(&mut self) -> u32 {
        self.state.0.0 = self.state.0.0.wrapping_add(0x9E3779B9);

        let mut z = self.state.0.0 as u64;
        z = (z ^ (z >> 16)).wrapping_mul(A);
        z = (z ^ (z >> 16)).wrapping_mul(B);
        (z ^ (z >> 16)) as u32
    }
}

impl Rng for SplitMix32 {
    type Word = u32;

    #[inline]
    fn nextu(&mut self) -> Self::Word {
        self.state += 0x9E3779B9;

        let mut z = self.state.cast::<u64>();
        z = (z ^ (z >> 16)) * A;
        z = (z ^ (z >> 16)) * B;
        *(z ^ (z >> 16)).cast::<u32>()
    }
}

#[cfg(feature = "simd")]
pub use simd::*;

#[cfg(feature = "simd")]
pub mod simd {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[cfg(target_arch = "x86_64")]
    use crate::VRng;

    pub const SPLITMIX32X16: usize = 16;
    pub const SPLITMIX32X16_PAR_CHUNK: usize = 8192;
    pub const SPLITMIX32_GAMMA: u32 = 0x9E37_79B9;

    /// 16-way SIMD implementation of SplitMix32 32-bit RNG.
    /// This implementation uses AVX-512F instructions to generate 16 random numbers in parallel.
    ///
    /// # Example
    /// ```no_run
    /// use urng::{VRng, SplitMix32x16};
    /// unsafe {
    ///     let mut rng = SplitMix32x16::new(1);
    ///     let _ = rng.nextuv();
    /// }
    /// ```
    #[cfg(target_arch = "x86_64")]
    #[repr(C, align(64))]
    pub struct SplitMix32x16 {
        pub(crate) state: __m512i,
    }

    #[cfg(target_arch = "x86_64")]
    impl SplitMix32x16 {
        /// Creates a new `SplitMix32x16` instance with the given seed.
        ///
        /// # Safety
        ///
        /// The caller must ensure the CPU supports the `avx512f` target feature.
        #[target_feature(enable = "avx512f")]
        pub unsafe fn new(seed: u32) -> Self {
            let base = seed | 1;
            let mut init = [0u32; SPLITMIX32X16];
            for (i, v) in init.iter_mut().enumerate() {
                *v = base.wrapping_add(SPLITMIX32_GAMMA.wrapping_mul((i as u32).wrapping_add(1)));
            }
            Self {
                state: unsafe { _mm512_loadu_si512(init.as_ptr() as *const _) },
            }
        }

        /// Computes the SplitMix32 output for 16 lanes at once.
        ///
        /// # Safety
        ///
        /// The caller must ensure the CPU supports the `avx512f` target feature.
        #[target_feature(enable = "avx512f")]
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
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    impl VRng for SplitMix32x16 {
        type Word = __m512i;

        fn nextuv(&mut self) -> __m512i {
            unsafe {
                let v = Self::compute(self.state);
                self.state = _mm512_add_epi32(
                    self.state,
                    _mm512_set1_epi32(SPLITMIX32_GAMMA.wrapping_mul(SPLITMIX32X16 as u32) as i32),
                );
                v
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! { SplitMix32 }
}
