use crate::rng::Rng64;
use crate::rng64::SplitMix64;
use crate::wrap;
use std::num::Wrapping;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A xoshiro256++ random number generator.
///
/// This is an all-purpose generator with 256-bit state.
/// It is particularly suitable for generating floating-point numbers.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xoshiro256Pp::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoshiro256Pp {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Pp {
    /// Creates a new `Xoshiro256Pp` instance seeded via `SplitMix64`.
    ///
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
        }
    }
}

impl Rng64 for Xoshiro256Pp {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let s = &mut self.s;
        let res = wrap!((s[0] + s[3]).0.rotate_left(23)) + s[0];
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = wrap!(s[3].0.rotate_left(45));

        res.0
    }
}

/// AVX-512 vectorized xoshiro256++ variant producing 2 `u64` values per call.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Xoshiro256Ssx2 {
    pub(crate) s: __m512i,
}

impl Xoshiro256Ssx2 {
    /// Creates a new `Xoshiro256Ssx2` instance.
    #[cfg(target_arch = "x86_64")]
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);

        let mut s = [0u64; 8];
        s.iter_mut().for_each(|v| *v = seedgen.nextu());

        unsafe {
            Self {
                s: _mm512_loadu_si512(s.as_ptr() as *const __m512i),
            }
        }
    }

    /// Generates the next 2 random `u64` values.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn nextu(&mut self) -> [u64; 2] {
        let s = &mut self.s;
        unsafe {
            // let res = wrap!((s[0] + s[3]).0.rotate_left(23)) + s[0];
            let res = _mm512_add_epi64(
                _mm512_rol_epi64(
                    _mm512_add_epi64(*s, _mm512_shuffle_epi32(*s, 0b11_10_01_00)),
                    23,
                ),
                *s,
            );
            // let t = s[1] << 17;
            let t = _mm512_slli_epi64(_mm512_shuffle_epi32(*s, 0b01_00_11_10), 17);

            // s[2] ^= s[0];
            *s = _mm512_xor_si512(*s, _mm512_shuffle_epi32(*s, 0b10_11_00_01));
            // s[3] ^= s[1];
            *s = _mm512_xor_si512(*s, _mm512_shuffle_epi32(*s, 0b11_10_01_00));
            // s[1] ^= s[2];
            *s = _mm512_xor_si512(*s, _mm512_shuffle_epi32(*s, 0b00_01_10_11));
            // s[0] ^= s[3];
            *s = _mm512_xor_si512(*s, _mm512_shuffle_epi32(*s, 0b01_00_11_10));

            // s[2] ^= t;
            *s = _mm512_xor_si512(*s, t);
            // s[3] = wrap!(s[3].0.rotate_left(45));
            *s = _mm512_rol_epi64(*s, 45);

            // res.0
            let mut out = [0u64; 2];
            _mm512_stream_si512(out.as_mut_ptr() as *mut __m512i, res);
            out
        }
    }
}

/// A xoshiro256** random number generator.
///
/// This is an all-purpose generator with 256-bit state.
/// It is robust against linear artifacts and generally recommended for all purposes.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Xoshiro256Ss::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoshiro256Ss {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Ss {
    /// Creates a new `Xoshiro256Ss` instance seeded via `SplitMix64`.
    ///
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
        }
    }
}

impl Rng64 for Xoshiro256Ss {
    #[inline]
    fn nextu(&mut self) -> u64 {
        let s = &mut self.s;
        let res = wrap!((s[1] * wrap!(5)).0.rotate_left(7)) * wrap!(9);
        let t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = wrap!(s[3].0.rotate_left(45));

        res.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xoshiro256Pp);
    crate::safe_test!(Xoshiro256Ss);
}
