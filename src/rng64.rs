use crate::rng::Rng64;
use crate::wrap;
use std::hint::black_box;
use std::num::Wrapping;
use wide::u32x4;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- Mt1993764 ---

/// A 64-bit Mersenne Twister (MT19937-64) random number generator.
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

// --- TwistedGFSR ---

/// A Twisted Generalized Feedback Shift Register (TGFSR) generator.
#[repr(C, align(64))]
pub struct TwistedGFSR {
    seed: [u64; N_GFSR],
    index: usize,
}

const N_GFSR: usize = 25;
const M_GFSR: usize = 7;

impl TwistedGFSR {
    /// Provides a default seed array.
    pub const fn new_seed() -> [u64; N_GFSR] {
        [
            0x95f24dab, 0x0b685215, 0xe76ccae7, 0xaf3ec239, 0x715fad23, 0x24a590ad, 0x69e4b5ef,
            0xbf456141, 0x96bc1b7b, 0xa7bdf825, 0xc1de75b7, 0x8858a9c9, 0x2da87693, 0xb657f9dd,
            0xffdc8a9f, 0x8121da71, 0x8b823ecb, 0x885d05f5, 0x4e20cd47, 0x5a9ad5d9, 0x512c0c03,
            0xea857ccd, 0x4cc1d30f, 0x8891a8a1, 0xa6b7aadb,
        ]
    }
    const fn mag01() -> [u64; 2] {
        [0x0, 0x8ebfd028]
    }

    /// Creates a new `TwistedGFSR` instance.
    pub fn new(seed: [u64; N_GFSR]) -> Self {
        Self {
            seed,
            index: N_GFSR,
        }
    }

    fn twist(&mut self) {
        for k in 0..(N_GFSR - M_GFSR) {
            self.seed[k] = self.seed[k + M_GFSR]
                ^ (self.seed[k] >> 1)
                ^ Self::mag01()[(self.seed[k] & 1) as usize];
        }
        for k in (N_GFSR - M_GFSR)..N_GFSR {
            self.seed[k] = self.seed[k + M_GFSR - N_GFSR]
                ^ (self.seed[k] >> 1)
                ^ Self::mag01()[(self.seed[k] & 1) as usize];
        }
        self.index = 0;
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        if self.index >= N_GFSR {
            self.twist();
        }
        let mut y = self.seed[self.index];
        y ^= (y << 7) & 0x2b5b2500;
        y ^= (y << 15) & 0xdb8b0000;
        y &= 0xffffffff;
        y ^= y >> 16;
        self.index += 1;
        y
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / 4294967296.0)
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

impl Rng64 for TwistedGFSR {
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

// --- Lcg64 ---

/// A Linear Congruential Generator (LCG) for 64-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
#[repr(C, align(64))]
#[deprecated(since = "0.2.4", note = "Use Xoshiro256++/** instead.")]
pub struct Lcg64 {
    x: Wrapping<u64>,
    a: u64,
    b: u64,
    m: u64,
    r: f64,
}

#[allow(deprecated)]
impl Lcg64 {
    /// Creates a new `Lcg64` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - The initial state (seed).
    /// * `a` - The multiplier.
    /// * `b` - The increment.
    /// * `m` - The modulus.
    /// * `warm` - The number of initial iterations to skip.
    pub fn new(x: u64, a: u64, b: u64, m: u64, warm: usize) -> Self {
        let a = a | 1;
        let mut rng = Self {
            x: wrap!(x),
            a,
            b,
            m,
            r: 1.0 / (m as f64 + 1.0),
        };

        (0..warm).into_iter().for_each(|_| {
            black_box(rng.nextu());
        });

        rng
    }

    /// Generates the next raw `u64` value via the LCG recurrence.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    /// Generates the next `f64` value in `[0, 1)`.
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * self.r
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
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

#[allow(deprecated)]
impl Rng64 for Lcg64 {
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

// --- Philox64 ---

/// A Philox 2x64 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[repr(C, align(64))]
pub struct Philox64 {
    pub(crate) c: [u64; 2],
    pub(crate) k: [u64; 2],
}

impl Philox64 {
    const fn m0() -> u128 {
        0xD2B74407B1CE6E93
    }

    /// Creates a new `Philox64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            c: [1, 0],
            k: [seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline]
    pub(crate) fn compute(c: [u64; 2], k: [u64; 2]) -> [u64; 2] {
        let mut v0 = c[0];
        let mut v1 = c[1];
        let mut key = k[0];

        let w0: u64 = 0x9E3779B97F4A7C15;

        for _ in 0..10 {
            let prod = (v0 as u128).wrapping_mul(Self::m0());
            let hi = (prod >> 64) as u64;
            let lo = prod as u64;
            let next_v0 = hi ^ v1 ^ key;
            let next_v1 = lo;

            v0 = next_v0;
            v1 = next_v1;
            key = key.wrapping_add(w0);
        }

        [v0, v1]
    }

    /// Generates the next block of random numbers.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 2] {
        let out = Self::compute(self.c, self.k);
        self.c[0] = self.c[0].wrapping_add(1);
        if self.c[0] == 0 {
            self.c[1] = self.c[1].wrapping_add(1);
        }
        out
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu()[0] as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu()[0];
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu()[0] as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1) as usize;
        &choices[index]
    }
}

impl Rng64 for Philox64 {
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

// --- Sfc64 ---

/// A 64-bit SFC random number generator.
///
/// All hot-path methods use `#[inline(always)]` to ensure the 4 state variables
/// (a, b, c, counter) remain pinned in CPU registers throughout batch loops.
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
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Sfc64x4 {
    a: __m256i,
    b: __m256i,
    c: __m256i,
    counter: __m256i,
}

#[cfg(target_arch = "x86_64")]
impl Sfc64x4 {
    /// Creates a new `Sfc64x4` from 4 independent seeds.
    ///
    /// # Safety
    /// Requires AVX2 support (guaranteed by `target-cpu=native` on modern x86_64).
    #[inline(always)]
    pub unsafe fn new(seeds: [u64; 4]) -> Self {
        let mut a = [0u64; 4];
        let mut b = [0u64; 4];
        let mut c = [0u64; 4];
        for i in 0..4 {
            let mut sg = SplitMix64::new(seeds[i]);
            a[i] = sg.nextu();
            b[i] = sg.nextu();
            c[i] = sg.nextu();
        }
        unsafe {
            Self {
                a: _mm256_loadu_si256(a.as_ptr() as *const __m256i),
                b: _mm256_loadu_si256(b.as_ptr() as *const __m256i),
                c: _mm256_loadu_si256(c.as_ptr() as *const __m256i),
                counter: _mm256_set1_epi64x(1),
            }
        }
    }

    /// Generates 4 random `u64` values simultaneously and writes them to `out`.
    ///
    /// # Safety
    /// `out` must point to a valid buffer of at least 4 `u64` values.
    /// Requires AVX2 support.
    #[inline(always)]
    pub unsafe fn next4u(&mut self, out: *mut u64) {
        unsafe {
            let one = _mm256_set1_epi64x(1);

            // res = a + b + counter
            let res = _mm256_add_epi64(_mm256_add_epi64(self.a, self.b), self.counter);

            // a = b ^ (b >> 11)
            self.a = _mm256_xor_si256(self.b, _mm256_srli_epi64(self.b, 11));

            // b = c + (c << 3)
            self.b = _mm256_add_epi64(self.c, _mm256_slli_epi64(self.c, 3));

            // c = rotate_left(res, 24) = (res << 24) | (res >> 40)
            self.c = _mm256_or_si256(_mm256_slli_epi64(res, 24), _mm256_srli_epi64(res, 40));

            // counter += 1
            self.counter = _mm256_add_epi64(self.counter, one);

            // Store 4 results
            _mm256_storeu_si256(out as *mut __m256i, res);
        }
    }

    /// Generates 4 random `f64` values in [0, 1) and writes them to `out`.
    #[inline(always)]
    pub unsafe fn next4f(&mut self, out: *mut f64) {
        unsafe {
            let mut buf = [0u64; 4];
            self.next4u(buf.as_mut_ptr());
            const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
            for i in 0..4 {
                *out.add(i) = buf[i] as f64 * SCALE;
            }
        }
    }

    /// Generates 4 random `i64` values in [min, max] and writes them to `out`.
    #[inline(always)]
    pub unsafe fn next4i(&mut self, out: *mut i64, min: i64, max: i64) {
        unsafe {
            let mut buf = [0u64; 4];
            self.next4u(buf.as_mut_ptr());
            let range = (max as i128 - min as i128 + 1) as u128;
            for i in 0..4 {
                *out.add(i) = ((buf[i] as u128 * range) >> 64) as i64 + min;
            }
        }
    }

    /// Generates 4 random `f64` values in [min, max) and writes them to `out`.
    #[inline(always)]
    pub unsafe fn next4rf(&mut self, out: *mut f64, min: f64, max: f64) {
        unsafe {
            let mut buf = [0u64; 4];
            self.next4u(buf.as_mut_ptr());
            let range = max - min;
            let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
            for i in 0..4 {
                *out.add(i) = buf[i] as f64 * scale + min;
            }
        }
    }
}

// --- Xorshift64 ---

/// A 64-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
#[repr(C)]
pub struct Xorshift64 {
    a: u64,
}

impl Xorshift64 {
    /// Creates a new `Xorshift64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self { a: seedgen.nextu() }
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let mut x = self.a;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.a = x;
        x
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

impl Rng64 for Xorshift64 {
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

/// A 64-bit Cellular Automata-based random number generator.
///
/// This generator uses a 4-cell cellular automaton state and a Weyl counter.
/// It is designed for high performance and quality.
#[repr(C)]
pub struct Cet64 {
    k: Wrapping<u64>,
    v: [Wrapping<u64>; 4],
    c: Wrapping<u64>,
}

impl Cet64 {
    /// Creates a new `Cet64` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let k = seedgen.nextu();
        Self {
            k: wrap!(k),
            v: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
            c: wrap!(1327),
        }
    }

    /// Generates the next random `u64` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::Cet64;
    ///
    /// let mut rng = Cet64::new(1);
    /// assert_eq!(rng.nextu(), 15169567334506313593);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// let f = rng.randf(0.0, 1.0);
    /// assert!(f >= 0.0 && f < 1.0);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let [mut a, mut b, mut c, mut d] = self.v;

        a += a.0.rotate_left(13).wrapping_mul(self.k.0);
        c += c.0.rotate_left(27) ^ self.c.0;

        b ^= a.0.rotate_left(32);
        d ^= c.0.rotate_left(32);

        self.c += 1327;
        self.v = [a, b, c, d];

        (((a ^ b) ^ (c ^ d)) ^ wrap!(182)).0
    }

    /// Generates the next `f64` value in `[0, 1)`.
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
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

impl Rng64 for Cet64 {
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

/// A SplitMix64 random number generator.
///
/// This is a fast generator with 64-bit state, often used for initializing
/// other generators from a single seed.
///
/// # Examples
///
/// ```
/// use urng::rng64::SplitMix64;
///
/// let mut rng = SplitMix64::new(1);
/// assert_eq!(rng.nextu(), 10451216379200822465);
/// ```
#[repr(align(64))]
pub struct SplitMix64 {
    pub(crate) s: Wrapping<u64>,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng64::SplitMix64;
    ///
    /// let mut rng = SplitMix64::new(1);
    /// assert_eq!(rng.nextu(), 10451216379200822465);
    /// assert_eq!(rng.nextf(), 0.7457817572627012);
    /// let i = rng.randi(0, 100);
    /// assert!(i >= 0 && i <= 100);
    /// ```
    pub fn new(seed: u64) -> Self {
        Self { s: wrap!(seed | 1) }
    }

    /// Computes the SplitMix64 output for a given raw state word (pure, stateless).
    #[inline]
    pub fn compute(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        self.s += 0x9E3779B97F4A7C15;
        Self::compute(self.s.0)
    }

    /// Generates the next `f64` value in `[0, 1)`.
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    /// Generates a random `i64` value in the range `[min, max]`.
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range `[min, max)`.
    #[inline]
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

impl Rng64 for SplitMix64 {
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

// --- Threefish256 ---

const THREEFISH_NW: usize = 4;
const THREEFISH_C240: u64 = 0x1BD11BDAA9FC1A22;
const THREE_FISH_N_ROUNDS: usize = 72;
const THREEFISH_PI: [usize; 4] = [0, 3, 2, 1];
const THREEFISH_R_256: [[u32; 2]; 8] = [
    [14, 16],
    [52, 57],
    [23, 40],
    [5, 37],
    [25, 33],
    [46, 12],
    [58, 22],
    [32, 32],
];

/// A Threefish-256 random number generator.
#[repr(C, align(64))]
pub struct Threefish256 {
    c: [u64; 4],
    k: [u64; 5],
    tw: [u64; 3],
    index: usize,
    buffer: [u64; 4],
}

impl Threefish256 {
    /// Creates a new `Threefish256` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let mut k = [0u64; 5];
        for i in 0..4 {
            k[i] = seedgen.nextu();
        }
        k[4] = THREEFISH_C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3];

        let tw0 = seedgen.nextu();
        let tw1 = seedgen.nextu();
        let tw = [tw0, tw1, tw0 ^ tw1];

        Self {
            c: [0; 4],
            k,
            tw,
            index: 4,
            buffer: [0; 4],
        }
    }

    #[inline(always)]
    fn mix(x0: u64, x1: u64, r: u32) -> [u64; 2] {
        let y0 = x0.wrapping_add(x1);
        [y0, x1.rotate_left(r) ^ y0]
    }

    #[inline(always)]
    fn key_schedule(k: &[u64; 5], tw: &[u64; 3], s: usize) -> [u64; 4] {
        [
            k[s % (THREEFISH_NW + 1)],
            k[(s + 1) % (THREEFISH_NW + 1)].wrapping_add(tw[s % 3]),
            k[(s + 2) % (THREEFISH_NW + 1)].wrapping_add(tw[(s + 1) % 3]),
            k[(s + 3) % (THREEFISH_NW + 1)].wrapping_add(s as u64),
        ]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u64; 4] {
        let mut v = self.c;

        for r in 0..THREE_FISH_N_ROUNDS {
            let mut e = [0u64; 4];
            if r % 4 == 0 {
                let ksi = Self::key_schedule(&self.k, &self.tw, r / 4);
                e[0] = v[0].wrapping_add(ksi[0]);
                e[1] = v[1].wrapping_add(ksi[1]);
                e[2] = v[2].wrapping_add(ksi[2]);
                e[3] = v[3].wrapping_add(ksi[3]);
            } else {
                e = v;
            }

            let mut f = [0u64; 4];
            let r_sh = THREEFISH_R_256[r % 8];
            let mx0 = Self::mix(e[0], e[1], r_sh[0]);
            f[0] = mx0[0];
            f[1] = mx0[1];
            let mx1 = Self::mix(e[2], e[3], r_sh[1]);
            f[2] = mx1[0];
            f[3] = mx1[1];

            v[0] = f[THREEFISH_PI[0]];
            v[1] = f[THREEFISH_PI[1]];
            v[2] = f[THREEFISH_PI[2]];
            v[3] = f[THREEFISH_PI[3]];
        }

        let ksi = Self::key_schedule(&self.k, &self.tw, THREE_FISH_N_ROUNDS.div_ceil(4));
        let mut dst = [0u64; 4];
        for i in 0..4 {
            dst[i] = v[i].wrapping_add(ksi[i]) ^ self.c[i];
        }

        self.c[0] = self.c[0].wrapping_add(1);
        if self.c[0] == 0 {
            self.c[1] = self.c[1].wrapping_add(1);
            if self.c[1] == 0 {
                self.c[2] = self.c[2].wrapping_add(1);
                if self.c[2] == 0 {
                    self.c[3] = self.c[3].wrapping_add(1);
                }
            }
        }

        dst
    }

    /// Generates the next random `u64` values.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 4] {
        if self.index >= 4 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 4;
        val
    }

    /// Generates the next random `f64` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f64; 4] {
        let out = self.nextu();
        const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
        let mut dst = [0f64; 4];
        for i in 0..4 {
            dst[i] = out[i] as f64 * SCALE;
        }
        dst
    }

    /// Generates random `i64` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> [i64; 4] {
        let range = (max as i128 - min as i128 + 1) as u128;
        let out = self.nextu();
        let mut dst = [0i64; 4];
        for i in 0..4 {
            dst[i] = ((out[i] as u128 * range) >> 64) as i64 + min;
        }
        dst
    }

    /// Generates random `f64` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> [f64; 4] {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        let out = self.nextu();
        let mut dst = [0f64; 4];
        for i in 0..4 {
            dst[i] = (out[i] as f64 * scale) + min;
        }
        dst
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

    #[test]
    fn twisted_gfsr_works() {
        let mut rng = TwistedGFSR::new(TwistedGFSR::new_seed());
        assert_eq!(rng.nextu(), 868393086);
        assert_eq!(rng.nextf(), 0.33567164628766477);
    }

    #[allow(deprecated)]
    #[test]
    fn lcg64_works() {
        let mut rng = Lcg64::new(8, 13, 5, 24, 0);
        assert_eq!(rng.nextu(), 13);
        assert_eq!(rng.nextf(), 0.24);
    }

    #[test]
    fn philox64_works() {
        let mut rng = Philox64::new(1);
        assert_eq!(rng.nextu(), [3996411588887038491, 2166702704631007519]);
        assert_eq!(rng.nextf(), 0.1488059942543676);
    }

    #[test]
    fn xorshift64_works() {
        let mut rng = Xorshift64::new(1);
        assert_eq!(rng.nextu(), 8247328468710148152);
        assert_eq!(rng.nextf(), 0.8223768786697171);
    }

    #[test]
    fn sfc64_works() {
        let mut rng = Sfc64::new(1);
        assert_eq!(rng.nextu(), 5761717516557699369);
        assert_eq!(rng.nextf(), 0.4850623141159338);
    }

    #[test]
    fn cet64_works() {
        let mut rng = Cet64::new(1);
        assert_eq!(rng.nextu(), 15169567334506313593);
        assert_eq!(rng.nextf(), 0.7143720878069354);
    }

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

    #[test]
    fn splitmix64_works() {
        let mut rng = SplitMix64::new(1);
        assert_eq!(rng.nextu(), 10451216379200822465);
        assert_eq!(rng.nextf(), 0.7457817572627012);
    }

    #[test]
    fn threefish256_works() {
        let mut rng = Threefish256::new(1);
        assert_eq!(
            rng.nextu(),
            [
                11703024954964515355,
                12040493508789228569,
                15247998991077977543,
                2489860152104538722
            ]
        );
        assert_eq!(
            rng.nextf(),
            [
                0.9622286493050641,
                0.5334118826690859,
                0.5452741654192154,
                0.6320415850102533
            ]
        );
    }
}
