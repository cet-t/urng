use crate::{rng::Rng32, rng64::SplitMix64, wrap};
use bytemuck::cast_slice;
use std::{hint::black_box, num::Wrapping};
use wide::u32x4;

use std::arch::x86_64::*;

// --- Mt19937 ---

/// A 32-bit Mersenne Twister (MT19937) random number generator.
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
    /// assert_eq!(rng.nextu(), 1811243163);
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
    /// assert_eq!(rng.nextu(), 1811243163);
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

// --- Lcg32 ---

/// A Linear Congruential Generator (LCG) for 32-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
#[repr(C)]
pub struct Lcg32 {
    x: Wrapping<u32>,
    a: u32,
    b: u32,
    m: u32,
    r: f32,
}

impl Lcg32 {
    /// Creates a new `Lcg32` instance.
    ///
    /// # Arguments
    ///
    /// * `x` - The initial state (seed).
    /// * `a` - The multiplier.
    /// * `b` - The increment.
    /// * `m` - The modulus.
    /// * `warm` - The number of initial iterations to skip (warm-up).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24, 0);
    /// assert_eq!(rng.nextu(), 13);
    /// ```
    pub fn new(x: u32, a: u32, b: u32, m: u32, warm: usize) -> Self {
        // M>a, M>b, A>0, B>=0
        let a = a | 1;
        let mut rng = Self {
            x: wrap!(x),
            a,
            b,
            m,
            r: 1.0 / (m as f32 + 1.0),
        };

        (0..warm).into_iter().for_each(|_| {
            black_box(rng.nextu());
        });

        rng
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24, 0);
    /// assert_eq!(rng.nextu(), 13);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        // X(n+1) = (a * X(n) + b) % M
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * self.r
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24, 0);
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
    /// use urng::rng32::Lcg32;
    ///
    /// let mut rng = Lcg32::new(8, 13, 5, 24, 0);
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

impl Rng32 for Lcg32 {
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

// --- Pcg32 ---

/// A PCG (Permuted Congruential Generator) random number generator.
///
/// This implementation uses the PCG-XSH-RR algorithm with 64-bit state and 32-bit output.
#[repr(C)]
pub struct Pcg32 {
    state: Wrapping<u64>,
    inc: Wrapping<u64>,
}

impl Pcg32 {
    /// Creates a new `Pcg32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Pcg32;
    ///
    /// let mut rng = Pcg32::new(1);
    /// assert_eq!(rng.nextu(), 1299482704);
    /// ```
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Pcg32 {
            state: wrap!(seedgen.nextu()),
            inc: wrap!(seedgen.nextu()),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Pcg32;
    ///
    /// let mut rng = Pcg32::new(1);
    /// assert_eq!(rng.nextu(), 1299482704);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate * wrap!(6364136223846793005) + self.inc;
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27).0;
        let rot = (oldstate >> 59).0;
        ((xorshifted >> rot) | (xorshifted << (rot.wrapping_neg() & 31))) as u32
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
    /// use urng::rng32::Pcg32;
    ///
    /// let mut rng = Pcg32::new(1);
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
    /// use urng::rng32::Pcg32;
    ///
    /// let mut rng = Pcg32::new(1);
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

impl Rng32 for Pcg32 {
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

// --- Pcg32x8 (AVX-512) ---

pub(crate) const PCG32X8_LANE: usize = 8;
pub(crate) const PCG32X8_PAR_CHUNK: usize = 131_072;
pub(crate) const PCG32X8_PAR_CHUNK_BLOCKS: u64 = (PCG32X8_PAR_CHUNK / PCG32X8_LANE) as u64;
pub(crate) const PCG32_MULT: u64 = 6364136223846793005;

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
#[repr(C)]
pub struct Pcg32Simd([u8; 0]);

// --- Philox32 ---

/// A Philox 4x32 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[repr(C)]
pub struct Philox32x4 {
    pub(crate) c: [Wrapping<u32>; 4],
    pub(crate) k: [Wrapping<u32>; 2],
}

impl Philox32x4 {
    /// Creates a new `Philox32x4` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Philox32x4;
    ///
    /// let mut rng = Philox32x4::new(1);
    /// assert_eq!(rng.nextu(), [1606368191, 902838097, 1231688191, 2515046358]);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            c: wrap![
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
            ],
            k: wrap![seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline(always)]
    pub(crate) fn compute(c: [Wrapping<u32>; 4], k: [Wrapping<u32>; 2]) -> [u32; 4] {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            unsafe {
                let mut v_x = _mm_loadu_si128(c.as_ptr() as *const _);
                let mut v_k = _mm_set_epi32(0, k[1].0 as i32, 0, k[0].0 as i32);

                let v_m = _mm_set_epi32(0, 0xCD9E8D57u32 as i32, 0, 0xD2511F53u32 as i32);
                let v_w = _mm_set_epi32(0, 0xBB67AE85u32 as i32, 0, 0x9E3779B9u32 as i32);

                for _ in 0..10 {
                    let prod = _mm_mul_epu32(v_x, v_m);
                    let shuf = _mm_shuffle_epi32(prod, 0x1B);
                    let x_shift = _mm_srli_epi64(v_x, 32);

                    v_x = _mm_xor_si128(shuf, _mm_xor_si128(x_shift, v_k));
                    v_k = _mm_add_epi32(v_k, v_w);
                }

                let mut out = [0u32; 4];
                _mm_storeu_si128(out.as_mut_ptr() as *mut _, v_x);
                out
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mut x = [c[0].0, c[1].0, c[2].0, c[3].0];
            let mut key = [k[0].0, k[1].0];

            const M0: u64 = 0xD2511F53;
            const M1: u64 = 0xCD9E8D57;
            const W0: u32 = 0x9E3779B9;
            const W1: u32 = 0xBB67AE85;

            for _ in 0..10 {
                let prod0 = (x[0] as u64).wrapping_mul(M0);
                let hi0 = (prod0 >> 32) as u32;
                let lo0 = prod0 as u32;

                let prod1 = (x[2] as u64).wrapping_mul(M1);
                let hi1 = (prod1 >> 32) as u32;
                let lo1 = prod1 as u32;

                x[0] = hi1 ^ x[1] ^ key[0];
                x[1] = lo1;
                x[2] = hi0 ^ x[3] ^ key[1];
                x[3] = lo0;

                key[0] = key[0].wrapping_add(W0);
                key[1] = key[1].wrapping_add(W1);
            }
            x
        }
    }

    /// Generates the next block of random numbers.
    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; 4] {
        let out = Self::compute(self.c, self.k);
        self.c[0] += 1;
        if self.c[0].0 == 0 {
            self.c[1] += 1;
            if self.c[1].0 == 0 {
                self.c[2] += 1;
                if self.c[2].0 == 0 {
                    self.c[3] += 1;
                }
            }
        }
        out
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline(always)]
    pub fn nextf(&mut self) -> f32 {
        self.nextu()[0] as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline(always)]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu()[0];
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline(always)]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu()[0] as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline(always)]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Philox32x4 {
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

// --- Philox32x4-10 x4 ---

#[allow(non_upper_case_globals)]
pub(crate) const PHILOX32x16: usize = 16;
#[allow(non_upper_case_globals)]
pub(crate) const PHILOX32x4x4_PAR_CHUNK: usize = 131_072;
#[allow(non_upper_case_globals)]
pub(crate) const PHILOX32x4x4_CHUNK_RATIO: u128 = (PHILOX32x4x4_PAR_CHUNK / PHILOX32x16) as u128;
#[allow(non_upper_case_globals)]
pub(crate) const PHILOX32x4x4_SHIFT: u128 = PHILOX32x4x4_CHUNK_RATIO.trailing_zeros() as u128;
#[allow(non_upper_case_globals)]
pub(crate) const PHILOX32x16_SHIFT: usize = PHILOX32x16.trailing_zeros() as usize;

#[cfg(target_arch = "x86_64")]
/// A Philox 4x32x4 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Philox32x4x4 {
    pub(crate) c: __m512i,
    pub(crate) k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Philox32x4x4 {
    /// Creates a new `Philox32x4x4` instance seeded with the given value.
    /// Requires AVX-512F support.
    ///
    /// # Safety
    ///
    /// Must only be called on a CPU that supports AVX-512F.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use urng::rng32::Philox32x4x4;
    ///
    /// // requires avx512f
    /// let mut rng = unsafe { Philox32x4x4::new(1) };
    /// let vals = unsafe { rng.nextu() };
    /// assert!(vals[0] > 0);
    /// ```
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut c = [0u32; PHILOX32x16];
        let mut k = [0u32; PHILOX32x16];

        let mut seedgen = SplitMix32::new(seed);
        c.iter_mut().for_each(|c| *c = seedgen.nextu());

        // [k0, 0, k1, 0]
        (0..PHILOX32x16).step_by(4).for_each(|i| {
            k[i + 0] = seedgen.nextu();
            k[i + 1] = 0;
            k[i + 2] = seedgen.nextu();
            k[i + 3] = 0;
        });

        unsafe {
            Self {
                c: _mm512_loadu_si512(c.as_ptr() as *const _),
                k: _mm512_loadu_si512(k.as_ptr() as *const _),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    pub(crate) fn compute(&mut self) -> [u32; PHILOX32x16] {
        let mut x = self.c;
        let mut key = self.k;
        let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
        let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);

        for _ in 0..10 {
            // x0 * M0, x2 * M1 = [lo0, hi0, lo1, hi1]
            let prod = _mm512_mul_epu32(x, m);

            // shuffle -> [hi1, lo1, hi0, lo0]
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);

            // x >> 32 -> [x1, 0, x3, 0]
            let x_shift = _mm512_srli_epi64(x, 32);

            // x ^ x_shift ^ key
            x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, key));

            // key += w
            key = _mm512_add_epi32(key, w);
        }

        unsafe {
            let mut out = [0u32; PHILOX32x16];
            _mm512_storeu_si512(out.as_mut_ptr() as *mut _, x);
            out
        }
    }

    /// Generates the next block of random numbers.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; PHILOX32x16] {
        let out = self.compute();

        // increment counter
        // [1, 1, 1, 1, 1, 1, 1, 1]
        let one = _mm512_set1_epi64(1);

        // lower 64 bits (indices 0, 2, 4, 6) +1
        let next_c = _mm512_mask_add_epi64(self.c, 0x55, self.c, one);

        // check overflow of lower 64 bits
        let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
        let carry_mask = (eq_zero_mask & 0x55) << 1;

        if carry_mask != 0 {
            // if overflow, add to upper 64 bits
            self.c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        } else {
            self.c = next_c;
        }

        out
    }

    /// Generates 16 random `f32` values in the range [0, 1) using AVX-512.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextf(&mut self) -> [f32; PHILOX32x16] {
        /*
        let out = self.nextu();
        let mut dst = [0f32; PHILOX32x16];
        let scale = 1.0 / (u32::MAX as f32 + 1.0);
        for i in 0..PHILOX32x16 {
            dst[i] = (out[i] as f32) * scale;
        }
        dst
        */

        unsafe {
            let out = self.nextu();
            let v_u32 = _mm512_loadu_si512(out.as_ptr() as *const _);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
            let v_res = _mm512_mul_ps(v_f32, scale);

            let mut res = [0f32; PHILOX32x16];
            _mm512_storeu_ps(res.as_mut_ptr(), v_res);
            res
        }
    }

    /// Generates a random `i32` value in the range [min, max].
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randi(&mut self, min: i32, max: i32) -> [i32; PHILOX32x16] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = unsafe { self.nextu() };
        let mut dst = [0i32; PHILOX32x16];
        for i in 0..PHILOX32x16 {
            dst[i] = ((out[i] as u64 * range) >> 32) as i32 + min;
        }
        dst
    }

    /// Generates a random `f32` value in the range [min, max).
    #[target_feature(enable = "avx512f")]
    pub unsafe fn randf(&mut self, min: f32, max: f32) -> [f32; PHILOX32x16] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = unsafe { self.nextu() };
        let mut dst = [0f32; PHILOX32x16];
        for i in 0..PHILOX32x16 {
            dst[i] = (out[i] as f32 * scale) + min;
        }
        dst
    }
}

// -- Philox32 --

/// Opaque handle for the Philox32 RNG.
/// Dispatched at runtime to AVX-512 (`Philox32x4x4`) or scalar (`Philox32x4`) implementation.
#[repr(C)]
pub struct Philox32([u8; 0]);

// --- Xorshift32 ---

/// A 32-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
#[repr(C)]
pub struct Xorshift32 {
    a: Wrapping<u32>,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
    /// assert_eq!(rng.nextu(), 2270655301);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            a: wrap!(sm.nextu()),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
    /// assert_eq!(rng.nextu(), 2270655301);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let x = self.a;
        self.a = x ^ (x << 13);
        self.a ^= x >> 17;
        self.a ^= x << 5;
        self.a.0
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
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
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
    /// use urng::rng32::Xorshift32;
    ///
    /// let mut rng = Xorshift32::new(1);
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

impl Rng32 for Xorshift32 {
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

/// A XORWOW random number generator.
///
/// This generator combines a Xorshift-based algorithm with a Weyl sequence (linear counter).
/// It has a state of 192 bits (5 x 32-bit state + 32-bit counter).
/// This is the default generator used in NVIDIA cuRAND.
///
/// # Examples
///
/// ```
/// use urng::rng32::Xorwow;
///
/// let mut rng = Xorwow::new(1);
/// assert_eq!(rng.nextu(), 1365527255);
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [Wrapping<u32>; 5],
    c: Wrapping<u32>,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
    /// assert_eq!(rng.nextu(), 1365527255);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            x: wrap![sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu(), sm.nextu()],
            c: wrap!(sm.nextu()),
        }
    }

    /// Generates the next random `u32` value.
    pub fn nextu(&mut self) -> u32 {
        let mut t = self.x[4];

        let s = self.x[0];
        self.x[4] = self.x[3];
        self.x[3] = self.x[2];
        self.x[2] = self.x[1];
        self.x[1] = s;

        t ^= t >> 2;
        t ^= t << 1;
        t ^= s ^ (s << 4);
        self.x[0] = t;
        self.c += wrap!(362437);
        (t + self.c).0
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
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
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
    /// use urng::rng32::Xorwow;
    ///
    /// let mut rng = Xorwow::new(1);
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
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Xorwow {
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }

    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }

    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

/// A SplitMix32 pseudo-random number generator.
///
/// Fast 32-bit finalizer-based PRNG commonly used to seed other generators.
/// Uses a single 32-bit state word advanced by the golden-ratio constant.
#[repr(C)]
pub struct SplitMix32 {
    state: Wrapping<u32>,
}

impl SplitMix32 {
    /// Creates a new `SplitMix32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// assert_ne!(rng.nextu(), 0);
    /// ```
    pub fn new(seed: u32) -> Self {
        Self {
            state: wrap!(seed | 1),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
    /// assert_ne!(rng.nextu(), 0);
    /// ```
    pub fn nextu(&mut self) -> u32 {
        self.state = self.state + wrap!(0x9E3779B9);
        let mut z = self.state;
        z = (z ^ (z >> 16)) + wrap!(0x85ebca6b);
        z = (z ^ (z >> 13)) + wrap!(0xc2b2ae35);
        (z ^ (z >> 16)).0
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
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
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
    /// use urng::rng32::SplitMix32;
    ///
    /// let mut rng = SplitMix32::new(1);
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
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for SplitMix32 {
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }

    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }

    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[allow(non_upper_case_globals)]
pub(crate) const SPLITMIX32x16: usize = 16;
#[allow(non_upper_case_globals)]
pub(crate) const SPLITMIX32x16_PAR_CHUNK: usize = 8192;
pub(crate) const SPLITMIX32_GAMMA: u32 = 0x9E37_79B9;

/// AVX-512 implementation of SplitMix32 producing 16 values per step.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct SplitMix32x16 {
    pub(crate) state: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl SplitMix32x16 {
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let base = seed | 1;
        let mut init = [0u32; SPLITMIX32x16];
        for (i, v) in init.iter_mut().enumerate() {
            *v = base.wrapping_add(SPLITMIX32_GAMMA.wrapping_mul((i as u32).wrapping_add(1)));
        }
        Self {
            state: unsafe { _mm512_loadu_si512(init.as_ptr() as *const _) },
        }
    }

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

    #[target_feature(enable = "avx512f")]
    pub unsafe fn nextu(&mut self) -> [u32; SPLITMIX32x16] {
        let v = unsafe { Self::compute(self.state) };
        self.state = _mm512_add_epi32(
            self.state,
            _mm512_set1_epi32(SPLITMIX32_GAMMA.wrapping_mul(SPLITMIX32x16 as u32) as i32),
        );
        let mut out = [0u32; SPLITMIX32x16];
        unsafe { _mm512_storeu_si512(out.as_mut_ptr() as *mut _, v) };
        out
    }
}

// -- SplitMix32Simd --

/// Opaque handle for the SplitMix32 RNG.
/// Dispatched at runtime to AVX-512 (`SplitMix32x16`) or scalar (`SplitMix32`) implementation.
#[repr(C)]
pub struct SplitMix32Simd([u8; 0]);

// --- Threefry32x4 ---

const THREEFRY32_C240: u32 = 0x1BD11BDA;
/// A Threefry4x32 random number generator (Random123 family).
///
/// This is a counter-based RNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 4 output values per block.
#[repr(C, align(64))]
pub struct Threefry32x4 {
    pub(crate) c: [u32; 4],
    pub(crate) k: [u32; 5],
    pub(crate) tw: [u32; 3],
    pub(crate) index: usize,
    pub(crate) buffer: [u32; 4],
}

impl Threefry32x4 {
    /// Creates a new `Threefry32x4` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Threefry32x4;
    ///
    /// let mut rng = Threefry32x4::new(1);
    /// assert_eq!(rng.nextu(), [12519260, 3511377784, 3358857301, 2366592296]);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut k = [0u32; 5];
        for i in 0..4 {
            k[i] = seedgen.nextu();
        }
        k[4] = THREEFRY32_C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3];

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

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 4 output words.
    ///
    /// # Arguments
    ///
    /// * `c`  - 4-word counter (the plaintext block).
    /// * `k`  - 5-word key schedule (k[4] = parity word).
    /// * `tw` - 3-word tweak schedule (tw[2] = tw[0] ^ tw[1]).
    #[inline(always)]
    pub fn compute(c: [u32; 4], k: &[u32; 5], tw: &[u32; 3]) -> [u32; 4] {
        let mut v = c;

        macro_rules! round {
            ($r_sh_0:expr, $r_sh_1:expr) => {
                let y0 = v[0].wrapping_add(v[1]);
                let f1 = v[1].rotate_left($r_sh_0) ^ y0;

                let y1 = v[2].wrapping_add(v[3]);
                let f3 = v[3].rotate_left($r_sh_1) ^ y1;

                v[0] = y0;
                v[1] = f3;
                v[2] = y1;
                v[3] = f1;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] = v[0].wrapping_add(k[$s % 5]);
                v[1] = v[1].wrapping_add(k[($s + 1) % 5].wrapping_add(tw[$s % 3]));
                v[2] = v[2].wrapping_add(k[($s + 2) % 5].wrapping_add(tw[($s + 1) % 3]));
                v[3] = v[3].wrapping_add(k[($s + 3) % 5].wrapping_add($s as u32));
            };
        }

        inject_key!(0);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(1);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(2);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(3);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(4);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        let ksi5_0 = k[0];
        let ksi5_1 = k[1].wrapping_add(tw[2]);
        let ksi5_2 = k[2].wrapping_add(tw[0]);
        let ksi5_3 = k[3].wrapping_add(5);

        [
            v[0].wrapping_add(ksi5_0) ^ c[0],
            v[1].wrapping_add(ksi5_1) ^ c[1],
            v[2].wrapping_add(ksi5_2) ^ c[2],
            v[3].wrapping_add(ksi5_3) ^ c[3],
        ]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u32; 4] {
        let dst = Self::compute(self.c, &self.k, &self.tw);

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

    /// Generates the next 4 random `u32` values.
    #[inline]
    pub fn nextu(&mut self) -> [u32; 4] {
        if self.index >= 4 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 4;
        val
    }

    /// Generates the next 4 random `f32` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f32; 4] {
        let out = self.nextu();
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
        let mut dst = [0f32; 4];
        for i in 0..4 {
            dst[i] = out[i] as f32 * SCALE;
        }
        dst
    }

    /// Generates 4 random `i32` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> [i32; 4] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = self.nextu();
        let mut dst = [0i32; 4];
        for i in 0..4 {
            dst[i] = ((out[i] as u64 * range) >> 32) as i32 + min;
        }
        dst
    }

    /// Generates 4 random `f32` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> [f32; 4] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = self.nextu();
        let mut dst = [0f32; 4];
        for i in 0..4 {
            dst[i] = (out[i] as f32 * scale) + min;
        }
        dst
    }
}

// --- Threefry32x2 ---

/// A Threefry 2x32 random number generator (Random123 family).
///
/// Counter-based PRNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 2 output values per block.
pub struct Threefry32x2 {
    pub(crate) c: [u32; 2],
    pub(crate) k: [u32; 3],
    pub(crate) buffer: [u32; 2],
    pub(crate) index: usize,
}

impl Threefry32x2 {
    /// Creates a new `Threefry32x2` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Threefry32x2;
    ///
    /// let mut rng = Threefry32x2::new(1);
    /// assert_eq!(rng.nextu(), [1748843679, 2574680703]);
    /// ```
    #[inline]
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        let k0 = sm.nextu();
        let k1 = sm.nextu();
        let k2 = k0 ^ k1 ^ THREEFRY32_C240;

        Self {
            c: [0, 0],
            k: [k0, k1, k2],
            buffer: [0; 2],
            index: 2,
        }
    }

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 2 output words.
    ///
    /// # Arguments
    ///
    /// * `c` - 2-word counter (the plaintext block).
    /// * `k` - 3-word key schedule (k[2] = k[0] ^ k[1] ^ C240).
    #[inline(always)]
    pub(crate) fn compute(c: [u32; 2], k: &[u32; 3]) -> [u32; 2] {
        let mut v = c;

        macro_rules! round {
            ($r_sh:expr) => {
                let y = v[0].wrapping_add(v[1]);
                let f = v[1].rotate_left($r_sh) ^ y;
                v[0] = y;
                v[1] = f;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] = v[0].wrapping_add(k[$s % 3]);
                v[1] = v[1].wrapping_add(k[($s + 1) % 3].wrapping_add($s as u32));
            };
        }

        inject_key!(0);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(1);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(2);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(3);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(4);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        let ksi5_0 = k[2];
        let ksi5_1 = k[0].wrapping_add(5);

        [v[0].wrapping_add(ksi5_0), v[1].wrapping_add(ksi5_1)]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u32; 2] {
        let dst = Self::compute(self.c, &self.k);
        let (n_c0, overflow) = self.c[0].overflowing_add(1);
        self.c[0] = n_c0;
        if overflow {
            self.c[1] = self.c[1].wrapping_add(1);
        }
        dst
    }

    /// Generates the next 2 random `u32` values.
    #[inline]
    pub fn nextu(&mut self) -> [u32; 2] {
        if self.index >= 2 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 2;
        val
    }

    /// Generates the next 2 random `f32` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f32; 2] {
        let out = self.nextu();
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
        [out[0] as f32 * SCALE, out[1] as f32 * SCALE]
    }

    /// Generates 2 random `i32` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> [i32; 2] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = self.nextu();
        [
            ((out[0] as u64 * range) >> 32) as i32 + min,
            ((out[1] as u64 * range) >> 32) as i32 + min,
        ]
    }

    /// Generates 2 random `f32` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> [f32; 2] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = self.nextu();
        [(out[0] as f32 * scale) + min, (out[1] as f32 * scale) + min]
    }
}

// --- Squares32 ---

/// The Squares random number generator (32-bit output version by Bernard Widynski).
pub struct Squares32 {
    pub(crate) c: u64,
    pub(crate) k: u64,
}

impl Squares32 {
    /// Creates a new `Squares32` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Squares32;
    ///
    /// let mut rng = Squares32::new(1);
    /// assert_eq!(rng.nextu(), 1225738608);
    /// ```
    #[inline]
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed | 1);
        Self {
            c: 0,
            k: seedgen.nextu(),
        }
    }

    /// Core computation: 4 rounds of middle-square with counter.
    /// Takes pre-computed y = ctr*key and z = y + key to avoid
    /// redundant multiplication in batch scenarios.
    #[inline(always)]
    pub fn compute_yz(y: u64, z: u64) -> u32 {
        let mut x: u64;

        x = y.wrapping_mul(y).wrapping_add(y);
        x = x.rotate_left(32);

        x = x.wrapping_mul(x).wrapping_add(z);
        x = x.rotate_left(32);

        x = x.wrapping_mul(x).wrapping_add(y);
        x = x.rotate_left(32);

        (x.wrapping_mul(x).wrapping_add(z) >> 32) as u32
    }

    /// Convenience wrapper: compute from counter and key directly.
    #[inline(always)]
    pub fn compute(ctr: u64, key: u64) -> u32 {
        let y = ctr.wrapping_mul(key);
        let z = y.wrapping_add(key);
        Self::compute_yz(y, z)
    }

    /// Generates the next random `u32` value.
    #[inline(always)]
    pub fn nextu(&mut self) -> u32 {
        let out = Self::compute(self.c, self.k);
        self.c = self.c.wrapping_add(1);
        out
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline(always)]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline(always)]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline(always)]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }
}

// C-ABI exports for Squares32

#[allow(non_upper_case_globals)]
pub(crate) const SQUARES32x8: usize = 8;

/// A high-throughput Squares random number generator utilizing AVX-512 SIMD instructions.
/// This implementation processes 8 counters in parallel and is highly optimized with 4-way unrolling.
#[cfg(target_arch = "x86_64")]
#[repr(C)]
#[repr(align(64))]
pub struct Squares32x8 {
    /// 8 counters stored in a 512-bit SIMD register.
    pub c: __m512i,
    /// 8 keys stored in a 512-bit SIMD register.
    pub k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Squares32x8 {
    /// Creates a new `Squares32x8` instance from a 32-bit seed.
    /// The seed is used to initialize the counters and keys.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut k = [0u64; SQUARES32x8];
        let mut seedgen = SplitMix64::new(seed as u64 | 1);
        k.iter_mut().for_each(|v| {
            *v = seedgen.nextu();
        });

        unsafe {
            Self {
                c: _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
                k: _mm512_loadu_si512(k.as_ptr() as *const _),
            }
        }
    }

    /// Core computation: 4 rounds of middle-square with counter.
    /// Returns 8x u32 random values in the lower 32-bits of each 64-bit lane.
    ///
    /// # Arguments
    /// * `y` - Pre-computed y = ctr * key.
    /// * `z` - Pre-computed z = y + key.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn compute_yz(y: __m512i, z: __m512i) -> __m256i {
        let mut x = _mm512_add_epi64(_mm512_mullo_epi64(y, y), y);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), z);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), y);
        x = _mm512_or_si512(_mm512_slli_epi64(x, 32), _mm512_srli_epi64(x, 32));

        _mm512_cvtepi64_epi32(_mm512_srli_epi64(
            _mm512_add_epi64(_mm512_mullo_epi64(x, x), z),
            32,
        ))
    }

    /// Convenience wrapper to compute random values from counter and key directly.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub(crate) unsafe fn compute(c: __m512i, k: __m512i) -> __m256i {
        unsafe {
            let y = _mm512_mullo_epi64(c, k);
            let z = _mm512_add_epi64(y, k);
            Self::compute_yz(y, z)
        }
    }

    /// Generates 8 new `u32` random numbers.
    /// Increments the internal counters by 8.
    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn nextu(&mut self) -> [u32; SQUARES32x8] {
        unsafe {
            let v = Self::compute(self.c, self.k);
            self.c = _mm512_add_epi64(self.c, _mm512_set1_epi64(8));
            let mut out = [0u32; SQUARES32x8];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut _, v);
            out
        }
    }
}

// -- Squares32Simd --

/// Opaque handle for the Squares32 RNG.
/// Dispatched at runtime to AVX-512 (`Squares32x8`) or scalar (`Squares32`) implementation.
#[repr(C)]
pub struct Squares32Simd([u8; 0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_works() {
        let mut rng = Mt19937::new(1);
        assert_eq!(rng.nextu(), 1811243163);
        assert_eq!(rng.nextf(), 0.7382414);
    }

    #[test]
    fn sfmt19937_works() {
        let mut rng = Sfmt19937::new(1);
        assert_eq!(rng.nextu(), 2240536539);
        assert_eq!(rng.nextf(), 0.89096653);
    }

    #[test]
    fn lcg32_works() {
        let mut rng = Lcg32::new(8, 13, 5, 24, 0);
        assert_eq!(rng.nextu(), 13);
        assert_eq!(rng.nextf(), 0.24);
    }

    #[test]
    fn pcg32_works() {
        let mut rng = Pcg32::new(1);
        assert_eq!(rng.nextu(), 1299482704);
        assert_eq!(rng.nextf(), 0.7210574);
    }

    #[test]
    fn philox32_works() {
        let mut rng = Philox32x4::new(1);
        assert_eq!(rng.nextu(), [1606368191, 902838097, 1231688191, 2515046358]);
        assert_eq!(rng.nextf(), 0.5834115);
    }

    #[test]
    fn philox32x4x4_works() {
        unsafe {
            let mut rng = Philox32x4x4::new(1);
            assert_eq!(
                rng.nextu(),
                [
                    3433810671, 3908867097, 2181896131, 1964852980, 502764505, 3339643839,
                    845579800, 356287197, 2203086005, 970497114, 2053487157, 3627004578,
                    1765004304, 1367891752, 630877398, 2591301858
                ]
            );
            assert_eq!(
                rng.nextf(),
                [
                    0.74029744,
                    0.22862802,
                    0.8084649,
                    0.44234967,
                    0.21707518,
                    0.063766554,
                    0.7941085,
                    0.43958178,
                    0.6189914,
                    0.41019267,
                    0.08147346,
                    0.58526325,
                    0.33999366,
                    0.60349184,
                    0.52620786,
                    0.041621894
                ]
            );
        }
    }

    #[test]
    fn xorshift32_works() {
        let mut rng = Xorshift32::new(1);
        assert_eq!(rng.nextu(), 2270655301);
        assert_eq!(rng.nextf(), 0.5149226);
    }

    #[test]
    fn xorwow_works() {
        let mut rng = Xorwow::new(1);
        assert_eq!(rng.nextu(), 1365527255);
        assert_eq!(rng.nextf(), 0.45477358);
    }

    #[test]
    fn threefry32x4_works() {
        let mut rng = Threefry32x4::new(1);
        assert_eq!(rng.nextu(), [12519260, 3511377784, 3358857301, 2366592296]);
        assert_eq!(
            rng.nextf(),
            [0.19581375, 0.019083649, 0.8195202, 0.87931794]
        );
    }

    #[test]
    fn threefry32x2_works() {
        let mut rng = Threefry32x2::new(1);
        assert_eq!(rng.nextu(), [1748843679, 2574680703]);
        assert_eq!(rng.nextf(), [0.62456435, 0.346636]);
    }

    #[test]
    fn squares32_works() {
        let mut rng = Squares32::new(1);
        assert_eq!(rng.nextu(), 1225738608);
        assert_eq!(rng.nextf(), 0.9183048);
    }

    #[test]
    fn squares32x8_works() {
        unsafe {
            let mut rng = Squares32x8::new(1);
            assert_eq!(
                rng.nextu(),
                [
                    1225738608, 3081786017, 2002165410, 1518623550, 443612158, 1744152856,
                    1924491776, 1460635941
                ]
            );
        }
    }
}
