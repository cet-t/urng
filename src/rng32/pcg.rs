use crate::rng::Rng64;
use crate::{rng::Rng32, rng64::SplitMix64, wrap};
use std::num::Wrapping;

use std::arch::x86_64::*;

// --- Pcg32 ---

/// A PCG (Permuted Congruential Generator) random number generator.
///
/// This implementation uses the PCG-XSH-RR algorithm with 64-bit state and 32-bit output.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Pcg32::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Pcg32 {
    state: Wrapping<u64>,
    inc: Wrapping<u64>,
}

impl Pcg32 {
    /// Creates a new `Pcg32` instance seeded with the given value.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Pcg32 {
            state: wrap!(seedgen.nextu()),
            inc: wrap!(seedgen.nextu()),
        }
    }
}

impl Rng32 for Pcg32 {
    #[inline]
    fn nextu(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate * wrap!(6364136223846793005) + self.inc;
        let xorshifted = ((((oldstate >> 18) ^ oldstate) >> 27).0) as u32;
        let rot = ((oldstate >> 59).0) as u32;
        xorshifted.rotate_right(rot)
    }
}

// --- Pcg32x8 (AVX-512) ---

pub const PCG32X8_LANE: usize = 8;
pub const PCG32X8_PAR_CHUNK: usize = 131_072;
pub const PCG32X8_PAR_CHUNK_BLOCKS: u64 = (PCG32X8_PAR_CHUNK / PCG32X8_LANE) as u64;
pub const PCG32_MULT: u64 = 6364136223846793005;

/// AVX-512 implementation of PCG32 producing 8 values per step.
///
/// # Examples
///
/// ```
/// use urng::rng32::Pcg32x8;
///
/// unsafe {
///     let mut rng = Pcg32x8::new(1);
///     let _ = rng.nextu();
/// }
/// ```
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Pcg32x8 {
    pub(crate) state: __m512i,
    pub(crate) inc: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Pcg32x8 {
    /// Creates a new `Pcg32x8` instance with 8 independent PCG32 streams.
    /// Requires AVX-512F support.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);

        let mut state = [0u64; PCG32X8_LANE];
        state.iter_mut().for_each(|v| *v = seedgen.nextu());

        let mut inc = [0u64; PCG32X8_LANE];
        inc.iter_mut().for_each(|v| *v = seedgen.nextu());

        unsafe {
            Pcg32x8 {
                state: _mm512_loadu_si512(state.as_ptr() as *const _),
                inc: _mm512_loadu_si512(inc.as_ptr() as *const _),
            }
        }
    }

    /// Advances all 8 PCG32 streams and returns their outputs.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; PCG32X8_LANE] {
        let mult_lo = _mm512_set1_epi64(0x4C957F2D_i64);
        let mult_hi = _mm512_set1_epi64(0x5851F42D_i64);
        let mask32 = _mm512_set1_epi64(0xFFFFFFFF_i64);
        let out256 = Self::step_u32(&mut self.state, self.inc, mult_lo, mult_hi, mask32);
        let mut out = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, out256);
        out
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn step_u32(
        state: &mut __m512i,
        inc: __m512i,
        mult_lo: __m512i,
        mult_hi: __m512i,
        mask32: __m512i,
    ) -> __m256i {
        let oldstate = *state;
        let state_hi = _mm512_srli_epi64(oldstate, 32);
        let prod_lo = _mm512_mul_epu32(oldstate, mult_lo);
        let cross = _mm512_add_epi64(
            _mm512_mul_epu32(state_hi, mult_lo),
            _mm512_mul_epu32(oldstate, mult_hi),
        );
        *state = _mm512_add_epi64(_mm512_add_epi64(prod_lo, _mm512_slli_epi64(cross, 32)), inc);

        let xs = _mm512_srli_epi64(
            _mm512_xor_si512(_mm512_srli_epi64(oldstate, 18), oldstate),
            27,
        );
        let rot = _mm512_srli_epi64(oldstate, 59);
        let rotated = _mm512_rorv_epi32(_mm512_and_si512(xs, mask32), rot);
        _mm512_cvtepi64_epi32(rotated)
    }
}

// -- Pcg32Simd --

/// Opaque handle for the Pcg32 RNG.
/// Dispatched at runtime to AVX-512 (`Pcg32x8`) or scalar (`Pcg32`) implementation.
///
/// # Examples
///
/// ```
/// use urng::rng32::Pcg32Simd;
///
/// let _ = core::mem::size_of::<Pcg32Simd>();
/// ```
#[repr(C)]
pub struct Pcg32Simd([u8; 0]);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Pcg32);
}
