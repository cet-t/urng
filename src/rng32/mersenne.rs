use crate::rng32::SplitMix32;
use crate::{rng::Rng32, rng64::SplitMix64, wrap};
use bytemuck::cast_slice;
use std::num::Wrapping;
use wide::u32x4;

// --- Mt19937 ---

/// A 32-bit Mersenne Twister (MT19937) random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng32::Mt19937;
///
/// let mut rng = Mt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Mt19937 {
    mt: [Wrapping<u32>; MT32_N],
    mti: Wrapping<usize>,
}

const MT32_N: usize = 624;
const MT32_M: usize = 397;
const MT32_MATRIX_A: u32 = 0x9908B0DF;
const MT32_UPPER_MASK: u32 = 0x80000000;
const MT32_LOWER_MASK: u32 = 0x7FFFFFFF;

impl Mt19937 {
    /// Creates a new `Mt19937` instance seeded with the given value.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial seed value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// assert_eq!(rng.nextu(), 460915295);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut mt = [wrap!(0u32); MT32_N];
        let mut seedgen = SplitMix32::new(seed);
        mt[0] = wrap!(seedgen.nextu());
        for i in 1..MT32_N {
            let prev = mt[i - 1];
            mt[i] = wrap!(1812433253u32) * (prev ^ (prev >> 30)) + wrap!(i as u32);
        }
        Self {
            mt,
            mti: wrap!(MT32_N),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// assert_eq!(rng.nextu(), 460915295);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        if self.mti.0 >= MT32_N {
            self.twist();
        }
        let mut y = self.mt[self.mti.0];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7).0 & 0x9D2C5680;
        y ^= (y << 15).0 & 0xEFC60000;
        y ^= y >> 18;
        y.0
    }

    fn twist(&mut self) {
        for i in 0..MT32_N - MT32_M {
            let x = (self.mt[i].0 & MT32_UPPER_MASK) | (self.mt[i + 1].0 & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M] ^ wrap!(x_a);
        }
        for i in MT32_N - MT32_M..MT32_N - 1 {
            let x = (self.mt[i].0 & MT32_UPPER_MASK) | (self.mt[i + 1].0 & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M - MT32_N] ^ wrap!(x_a);
        }
        let x = (self.mt[MT32_N - 1].0 & MT32_UPPER_MASK) | (self.mt[0].0 & MT32_LOWER_MASK);
        let mut x_a = x >> 1;
        if x & 1 != 0 {
            x_a ^= MT32_MATRIX_A;
        }
        self.mt[MT32_N - 1] = self.mt[MT32_M - 1] ^ wrap!(x_a);
        self.mti = wrap!(0);
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
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
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
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
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
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Mt19937 {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

// --- Sfmt19937 ---

/// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator (32-bit version).
///
/// # Examples
///
/// ```
/// use urng::rng32::Sfmt19937;
///
/// let mut rng = Sfmt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
#[repr(align(16))]
pub struct Sfmt19937 {
    state: [u32x4; SFMT_N],
    idx: usize,
}

const SFMT_N: usize = 156;
const SFMT_POS1: usize = 122;
const SFMT_SL1: u32 = 18;
const SFMT_SR1: u32 = 11;

const SFMT_MSK1: u32 = 0xdfffffef;
const SFMT_MSK2: u32 = 0xddfecb7f;
const SFMT_MSK3: u32 = 0xbffaffff;
const SFMT_MSK4: u32 = 0xbffffff6;
const SFMT_PARITY1: u32 = 0x00000001;
const SFMT_PARITY2: u32 = 0x00000000;
const SFMT_PARITY3: u32 = 0x00000000;
const SFMT_PARITY4: u32 = 0x13c9e684;

impl Sfmt19937 {
    /// Creates a new `Sfmt19937` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// assert_eq!(rng.nextu(), 2240536539);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);

        // Initialize state using u32 array for simplicity
        let mut raw_state = [0u32; SFMT_N * 4];
        for i in 0..SFMT_N * 2 {
            // Fill with 64-bit values from SplitMix64
            let s = seedgen.nextu();
            raw_state[2 * i] = s as u32;
            raw_state[2 * i + 1] = (s >> 32) as u32;
        }

        let mut state = [u32x4::default(); SFMT_N];
        for i in 0..SFMT_N {
            state[i] = u32x4::from([
                raw_state[4 * i],
                raw_state[4 * i + 1],
                raw_state[4 * i + 2],
                raw_state[4 * i + 3],
            ]);
        }

        let mut rng = Self {
            state,
            idx: SFMT_N * 4, // Force generate on first call. 156 * 4 = 624 u32s
        };
        rng.period_certification();
        rng
    }

    fn gen_rand_all(&mut self) {
        unsafe {
            let ptr = self.state.as_mut_ptr();
            let mut r1 = *ptr.add(SFMT_N - 2);
            let mut r2 = *ptr.add(SFMT_N - 1);

            // Constant mask vector
            let mask = u32x4::from([SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4]);

            for i in 0..(SFMT_N - SFMT_POS1) {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1);

                // x = lshift128(a, SFMT_SL2=1)
                let x: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(a)) << 8);
                // y = rshift128(c=r1, SFMT_SR2=1)
                let y: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(r1)) >> 8);

                let r = a ^ x ^ ((b >> SFMT_SR1) & mask) ^ y ^ (r2 << SFMT_SL1);

                *p_i = r;
                r1 = r2;
                r2 = r;
            }

            for i in (SFMT_N - SFMT_POS1)..SFMT_N {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1 - SFMT_N);

                // x = lshift128(a, SFMT_SL2=1)
                let x: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(a)) << 8);
                // y = rshift128(c=r1, SFMT_SR2=1)
                let y: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(r1)) >> 8);

                let r = a ^ x ^ ((b >> SFMT_SR1) & mask) ^ y ^ (r2 << SFMT_SL1);

                *p_i = r;
                r1 = r2;
                r2 = r;
            }
        }
    }

    fn period_certification(&mut self) {
        let mut inner = 0;
        let psfmt32 =
            unsafe { std::slice::from_raw_parts(self.state.as_ptr() as *const u32, SFMT_N * 4) };
        let parity = [SFMT_PARITY1, SFMT_PARITY2, SFMT_PARITY3, SFMT_PARITY4];

        for i in 0..4 {
            inner ^= psfmt32[i] & parity[i];
        }
        let mut i = 16;
        while i > 0 {
            inner ^= inner >> i;
            i >>= 1;
        }
        inner &= 1;

        // Verification passed
        if inner == 1 {
            return;
        }

        // Modification for period certification
        let psfmt32_mut = unsafe {
            std::slice::from_raw_parts_mut(self.state.as_mut_ptr() as *mut u32, SFMT_N * 4)
        };

        for i in 0..4 {
            let mut work = 1;
            for _ in 0..32 {
                if (work & parity[i]) != 0 {
                    psfmt32_mut[i] ^= work;
                    return;
                }
                work <<= 1;
            }
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// assert_eq!(rng.nextu(), 2240536539);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        if self.idx >= SFMT_N * 4 {
            self.gen_rand_all();
            self.idx = 0;
        }

        let s: &[u32] = cast_slice(&self.state);
        let val = s[self.idx];
        self.idx += 1;
        val
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
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
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
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Sfmt19937 {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_works() {
        let mut rng = Mt19937::new(1);
        assert_eq!(rng.nextu(), 460915295);
        assert_eq!(rng.nextf(), 0.068138085);
    }

    #[test]
    fn sfmt19937_works() {
        let mut rng = Sfmt19937::new(1);
        assert_eq!(rng.nextu(), 2240536539);
        assert_eq!(rng.nextf(), 0.89096653);
    }
}
