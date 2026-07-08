#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

use wrapn::Wrap;

use crate::rng::Rng64;
use crate::rng64::SplitMix64;

// --- Sfc64 ---

/// A 64-bit SFC random number generator.
///
/// # Examples
///
/// ```
/// use urng::*;
///
/// let mut rng = Sfc64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc64 {
    a: Wrap<u64>,
    b: Wrap<u64>,
    c: Wrap<u64>,
    counter: Wrap<u64>,
}

impl Sfc64 {
    /// Creates a new `Sfc64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            a: seedgen.nextu().into(),
            b: seedgen.nextu().into(),
            c: seedgen.nextu().into(),
            counter: 1.into(),
        }
    }
}

impl Rng64 for Sfc64 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        let res = self.a + self.b + self.counter;
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c + (self.c << 3);
        self.c = res.rotate_left(24);

        self.counter += 1;
        res.value()
    }
}

// --- Sfc64x4 (AVX2) ---

/// A 4-way SIMD SFC64 generator using AVX2 256-bit intrinsics.
///
/// # Examples
///
/// ```no_run
/// use urng::rng64::Sfc64x8;
///
/// unsafe {
///     let mut rng = Sfc64x8::new(0);
///     let _ = rng.nextu();
/// }
/// ```
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[repr(C, align(64))]
pub struct Sfc64x8 {
    a: __m512i,
    b: __m512i,
    c: __m512i,
    counter: __m512i,
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
impl Sfc64x8 {
    /// Creates a new `Sfc64x8` from 8 independent seeds.
    ///
    /// # Safety
    /// Requires AVX512 support (guaranteed by `target-cpu=native` on modern x86_64).
    #[inline(always)]
    pub fn new(seed: u64) -> Self {
        let mut a = [0u64; 8];
        let mut b = [0u64; 8];
        let mut c = [0u64; 8];
        let mut sg = SplitMix64::new(seed);
        for i in 0..8 {
            a[i] = sg.nextu();
            b[i] = sg.nextu();
            c[i] = sg.nextu();
        }
        unsafe {
            Self {
                a: _mm512_loadu_si512(a.as_ptr() as *const __m512i),
                b: _mm512_loadu_si512(b.as_ptr() as *const __m512i),
                c: _mm512_loadu_si512(c.as_ptr() as *const __m512i),
                counter: _mm512_set1_epi64(1),
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
            let one = _mm512_set1_epi64(1);

            // res = a + b + counter
            let res = _mm512_add_epi64(_mm512_add_epi64(self.a, self.b), self.counter);

            // a = b ^ (b >> 11)
            self.a = _mm512_xor_si512(self.b, _mm512_srli_epi64(self.b, 11));

            // b = c + (c << 3)
            self.b = _mm512_add_epi64(self.c, _mm512_slli_epi64(self.c, 3));

            // c = rotate_left(res, 24) = (res << 24) | (res >> 40)
            self.c = _mm512_or_si512(_mm512_slli_epi64(res, 24), _mm512_srli_epi64(res, 40));

            // counter += 1
            self.counter = _mm512_add_epi64(self.counter, one);

            // Store 8 results
            let mut out = [0u64; 8];
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, res);
            out
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
            let scale = 1.0 / (u64::MAX as f64 + 1.0);
            for i in 0..8 {
                out[i] = u[i] as f64 * scale;
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

    crate::safe_test!(Sfc64);

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    crate::unsafe_test!(Sfc64x8);
}
