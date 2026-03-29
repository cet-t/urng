use crate::rng::Rng64;
use crate::rng64::SplitMix64;
use bytemuck;
use wide::u32x4;

// --- Mt1993764 ---

/// A 64-bit Mersenne Twister (MT19937-64) random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng64::Mt1993764;
///
/// let mut rng = Mt1993764::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Mt1993764 {
    mt: [u64; N],
    mti: usize,
}

const N: usize = 312;
const M: usize = 156;
const MATRIX_A: u64 = 0xB5026F5AA96619E9;
const UPPER_MASK: u64 = 0xFFFFFFFF80000000;
const LOWER_MASK: u64 = 0x7FFFFFFF;

impl Mt1993764 {
    /// Creates a new `Mt1993764` instance seeded via `SplitMix64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Mt1993764;
    ///
    /// let mut rng = Mt1993764::new(1);
    /// assert_eq!(rng.nextu(), 9822250072823399003);
    /// assert_eq!(rng.nextf(), 0.8926985632057756);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; N];
        let mut seedgen = SplitMix64::new(seed);
        mt[0] = seedgen.nextu();
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = 6364136223846793005u64
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, mti: N }
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= (y >> 29) & 0x5555555555555555;
        y ^= (y << 17) & 0x71D67FFFEDA60000;
        y ^= (y << 37) & 0xFFF7EEE000000000;
        y ^= y >> 43;
        y
    }

    fn twist(&mut self) {
        for i in 0..N - M {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.mt[i] = self.mt[i + M] ^ x_a;
        }
        for i in N - M..N - 1 {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.mt[i] = self.mt[i + M - N] ^ x_a;
        }
        let x = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        let mut x_a = x >> 1;
        if x & 1 != 0 {
            x_a ^= MATRIX_A;
        }
        self.mt[N - 1] = self.mt[M - 1] ^ x_a;
        self.mti = 0;
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Mt1993764 {
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

// --- Sfmt1993764 ---

/// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng64::Sfmt1993764;
///
/// let mut rng = Sfmt1993764::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfmt1993764 {
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

impl Sfmt1993764 {
    /// Creates a new `Sfmt1993764` instance seeded via `SplitMix64`.
    ///
    /// The state is period-certified after initialisation.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Sfmt1993764;
    ///
    /// let mut rng = Sfmt1993764::new(1);
    /// assert_eq!(rng.nextu(), 16435431249378271195);
    /// assert_eq!(rng.nextf(), 0.914246861393214);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
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
            idx: SFMT_N * 2, // Force generate on first call
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

                // do_recursion inlined
                // a=state[i], b=state[i+SFMT_POS1], c=r1, d=r2
                // r = a ^ x ^ ((b >> SFMT_SR1) & mask) ^ y ^ (d << SFMT_SL1)

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

                // Group u32x4 operations
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

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        if self.idx >= SFMT_N * 2 {
            self.gen_rand_all();
            self.idx = 0;
        }

        // Use cast_slice which was faster in previous benchmarks
        let s: &[u64] = bytemuck::cast_slice(&self.state);
        let val = s[self.idx];
        self.idx += 1;
        val
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for Sfmt1993764 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt1993764_works() {
        let mut rng = Mt1993764::new(1);
        assert_eq!(rng.nextu(), 9822250072823399003);
        assert_eq!(rng.nextf(), 0.8926985632057756);
    }

    #[test]
    fn sfmt_works() {
        let mut rng = Sfmt1993764::new(1);
        assert_eq!(rng.nextu(), 16435431249378271195);
        assert_eq!(rng.nextf(), 0.914246861393214);
    }
}
