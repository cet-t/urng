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
/// use urng::rng64::Xoshiro256Pp;
///
/// let mut rng = Xoshiro256Pp::new(1);
/// assert_eq!(rng.nextu(), 14971601782005023387);
/// ```
#[repr(C)]
pub struct Xoshiro256Pp {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Pp {
    /// Creates a new `Xoshiro256Pp` instance seeded via `SplitMix64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Xoshiro256Pp;
    ///
    /// let mut rng = Xoshiro256Pp::new(1);
    /// assert_eq!(rng.nextu(), 14971601782005023387);
    /// assert_eq!(rng.nextf(), 0.7471047161582187);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// ```
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

    /// Generates the next random `u64` value.
    pub fn nextu(&mut self) -> u64 {
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

    /// Generates the next `f64` value in `[0, 1)`.
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Xoshiro256Pp {
    #[inline]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Xoshiro256Ssx2 {
    pub(crate) s: __m512i,
}

impl Xoshiro256Ssx2 {
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
/// use urng::rng64::Xoshiro256Ss;
///
/// let mut rng = Xoshiro256Ss::new(1);
/// assert_eq!(rng.nextu(), 12966619160104079557);
/// ```
#[repr(C)]
pub struct Xoshiro256Ss {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Ss {
    /// Creates a new `Xoshiro256Ss` instance seeded via `SplitMix64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Xoshiro256Ss;
    ///
    /// let mut rng = Xoshiro256Ss::new(1);
    /// assert_eq!(rng.nextu(), 12966619160104079557);
    /// assert_eq!(rng.nextf(), 0.520436619938857);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// ```
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

    /// Generates the next random `u64` value.
    pub fn nextu(&mut self) -> u64 {
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

    /// Generates the next `f64` value in `[0, 1)`.
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Xoshiro256Ss {
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro256pp_works() {
        let mut rng = Xoshiro256Pp::new(1);
        assert_eq!(rng.nextu(), 14971601782005023387);
        assert_eq!(rng.nextf(), 0.7471047161582187);
    }

    #[test]
    fn xoshiro256ss_works() {
        let mut rng = Xoshiro256Ss::new(1);
        assert_eq!(rng.nextu(), 12966619160104079557);
        assert_eq!(rng.nextf(), 0.520436619938857)
    }
}
