use crate::rng::Rng64;
use crate::rng64::SplitMix64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- Sfc64 ---

/// A 64-bit SFC random number generator.
///
/// All hot-path methods use `#[inline(always)]` to ensure the 4 state variables
/// (a, b, c, counter) remain pinned in CPU registers throughout batch loops.
///
/// # Examples
///
/// ```
/// use urng::rng64::Sfc64;
///
/// let mut rng = Sfc64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl Sfc64 {
    /// Creates a new `Sfc64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            a: seedgen.nextu(),
            b: seedgen.nextu(),
            c: seedgen.nextu(),
            counter: 1,
        }
    }

    /// Generates the next random `u64` value.
    #[inline(always)]
    pub fn nextu(&mut self) -> u64 {
        let res = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = res.rotate_left(24);

        self.counter = self.counter.wrapping_add(1);
        res
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline(always)]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline(always)]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline(always)]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline(always)]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Sfc64 {
    #[inline(always)]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }
    #[inline(always)]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }
    #[inline(always)]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

// --- Sfc64x4 (AVX2) ---

/// A 4-way SIMD SFC64 generator using AVX2 256-bit intrinsics.
///
/// Packs 4 independent SFC64 states into `__m256i` registers.
/// Each `next4u()` call produces 4 random `u64` values simultaneously.
///
/// # Examples
///
/// ```ignore
/// use urng::rng64::Sfc64x4;
///
/// let mut rng = unsafe { Sfc64x4::new([1, 2, 3, 4]) };
/// let mut out = [0u64; 4];
/// unsafe { rng.next4u(out.as_mut_ptr()) };
/// assert_eq!(out.len(), 4);
/// ```
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Sfc64x8 {
    a: __m512i,
    b: __m512i,
    c: __m512i,
    counter: __m512i,
}

#[cfg(target_arch = "x86_64")]
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

    #[test]
    fn sfc64_works() {
        unsafe {
            let mut rng = Sfc64x8::new(1);
            assert_eq!(
                rng.nextu(),
                [
                    5761717516557699369,
                    16392217990948748997,
                    7386369014502375963,
                    3655015268200462072,
                    18170354754293727307,
                    14985574646059259295,
                    10441402005730553591,
                    10649205844500993130
                ]
            );
            assert_eq!(
                rng.nextf(),
                [
                    0.4850623141159338,
                    0.3102235603972498,
                    0.09287880555094546,
                    0.852976412114921,
                    0.4535304830821644,
                    0.9831747932946263,
                    0.47839713097547637,
                    0.6040453125379837
                ]
            );
        };
    }
}
