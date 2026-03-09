use crate::{dispatch_simd, rng::Rng32, rng64::SplitMix64, wrap};
use bytemuck::cast_slice;
use rayon::prelude::*;
use std::{hint::black_box, num::Wrapping, ptr::copy_nonoverlapping, slice::from_raw_parts_mut};
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
    /// Creates a new `Mt19937` instance.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial seed value.
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
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
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

#[unsafe(no_mangle)]
pub extern "C" fn mt19937_new(seed: u32) -> *mut Mt19937 {
    Box::into_raw(Box::new(Mt19937::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_free(ptr: *mut Mt19937) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_next_u32s(ptr: *mut Mt19937, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_next_f32s(ptr: *mut Mt19937, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_rand_i32s(
    ptr: *mut Mt19937,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_rand_f32s(
    ptr: *mut Mt19937,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
    /// Creates a new `Sfmt` instance.
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
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
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

#[unsafe(no_mangle)]
pub extern "C" fn sfmt_new(seed: u64) -> *mut Sfmt19937 {
    Box::into_raw(Box::new(Sfmt19937::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn sfmt_free(ptr: *mut Sfmt19937) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn sfmt_next_u32s(ptr: *mut Sfmt19937, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn sfmt_next_f32s(ptr: *mut Sfmt19937, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn sfmt_rand_i32s(
    ptr: *mut Sfmt19937,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn sfmt_rand_f32s(
    ptr: *mut Sfmt19937,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
    /// * `warm` - The number of initial iterations to skip.
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

    #[inline]
    pub fn nextu(&mut self) -> u32 {
        // X(n+1) = (a * X(n) + b) % M
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * self.r
    }

    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn lcg32_new(x: u32, a: u32, b: u32, m: u32, warm: usize) -> *mut Lcg32 {
    Box::into_raw(Box::new(Lcg32::new(x, a, b, m, warm)))
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_free(ptr: *mut Lcg32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_next_u32s(ptr: *mut Lcg32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_next_f32s(ptr: *mut Lcg32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_rand_i32s(
    ptr: *mut Lcg32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_rand_f32s(
    ptr: *mut Lcg32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
    /// Creates a new `Pcg32` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Pcg32 {
            state: wrap!(seedgen.nextu()),
            inc: wrap!(seedgen.nextu() | 1),
        }
    }

    /// Generates the next random `u32` value.
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
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
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

#[unsafe(no_mangle)]
pub extern "C" fn pcg32_new(seed: u64) -> *mut Pcg32 {
    Box::into_raw(Box::new(Pcg32::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_free(ptr: *mut Pcg32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_next_u32s(ptr: *mut Pcg32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_next_f32s(ptr: *mut Pcg32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_rand_i32s(
    ptr: *mut Pcg32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_rand_f32s(
    ptr: *mut Pcg32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
    }
}

// --- Philox32 ---

const PHILOX32_PAR_CHUNK: usize = 4096;

/// A Philox 4x32 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[repr(C)]
pub struct Philox32x4 {
    c: [Wrapping<u32>; 4],
    k: [Wrapping<u32>; 2],
}

impl Philox32x4 {
    /// Creates a new `Philox32` instance.
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
    fn compute(c: [Wrapping<u32>; 4], k: [Wrapping<u32>; 2]) -> [u32; 4] {
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

#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_new(seed: u32) -> *mut Philox32x4 {
    Box::into_raw(Box::new(Philox32x4::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_free(ptr: *mut Philox32x4) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_next_u32s(ptr: *mut Philox32x4, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    dst[2] = result[2];
                    dst[3] = result[3];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_next_f32s(ptr: *mut Philox32x4, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale = 1.0f32 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = result[0] as f32 * scale;
                    dst[1] = result[1] as f32 * scale;
                    dst[2] = result[2] as f32 * scale;
                    dst[3] = result[3] as f32 * scale;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * scale;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_rand_i32s(
    ptr: *mut Philox32x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((result[2] as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((result[3] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_rand_f32s(
    ptr: *mut Philox32x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    dst[2] = (result[2] as f32 * scale_val) * range_val + min;
                    dst[3] = (result[3] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}

// --- Philox32x4-10 x4 ---

#[allow(non_upper_case_globals)]
const PHILOX32x16: usize = 16;
#[allow(non_upper_case_globals)]
const PHILOX32x4x4_PAR_CHUNK: usize = 16384;
#[allow(non_upper_case_globals)]
const PHILOX32x4x4_CHUNK_RATIO: u128 = (PHILOX32x4x4_PAR_CHUNK / PHILOX32x16) as u128;
#[allow(non_upper_case_globals)]
const PHILOX32x4x4_SHIFT: u128 = PHILOX32x4x4_CHUNK_RATIO.trailing_zeros() as u128;
#[allow(non_upper_case_globals)]
const PHILOX32x16_SHIFT: usize = PHILOX32x16.trailing_zeros() as usize;

/// A Philox 4x32x4 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Philox32x4x4 {
    c: __m512i,
    k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Philox32x4x4 {
    /// Creates a new `Philox32x4x4` instance.
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

#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_new(seed: u32) -> *mut Philox32x4x4 {
    unsafe { Box::into_raw(Box::new(Philox32x4x4::new(seed))) }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_free(ptr: *mut Philox32x4x4) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_next_u32s(ptr: *mut Philox32x4x4, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_next_u32s_chunk(chunk_idx, chunk, c, k, one)
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_next_f32s(ptr: *mut Philox32x4x4, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c0 = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);
        let scale = _mm512_set1_ps(1.0f32 / (u32::MAX as f32 + 1.0));

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_next_f32s_chunk(chunk_idx, chunk, c0, k, one, scale);
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_rand_i32s(
    ptr: *mut Philox32x4x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);
        let range = (max as i64 - min as i64 + 1) as u64;

        let v_range = _mm512_set1_epi64(range as i64);
        let v_min = _mm512_set1_epi32(min);
        let merge_mask = 0b1010101010101010;

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_rand_i32s_chunk(
                    chunk_idx, chunk, c, k, one, v_range, v_min, merge_mask,
                );
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_rand_f32s(
    ptr: *mut Philox32x4x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);

        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;
        let scale_mul_range = scale_val * range_val;

        let v_mult = _mm512_set1_ps(scale_mul_range);
        let v_min = _mm512_set1_ps(min);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_rand_f32s_chunk(chunk_idx, chunk, c, k, one, v_mult, v_min);
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}

#[cfg(target_arch = "x86_64")]
pub type Philox32 = core::ffi::c_void;

#[cfg(not(target_arch = "x86_64"))]
pub type Philox32 = core::ffi::c_void;

#[unsafe(no_mangle)]
pub extern "C" fn philox32_new(seed: u32) -> *mut Philox32 {
    dispatch_simd!(Philox32, philox32x4_new, philox32x4x4_new, seed)
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_free(ptr: *mut Philox32) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_free,
        philox32x4x4_free,
        ptr
    )
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_u32s(ptr: *mut Philox32, out: *mut u32, count: usize) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_next_u32s,
        philox32x4x4_next_u32s,
        ptr,
        out,
        count
    )
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_f32s(ptr: *mut Philox32, out: *mut f32, count: usize) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_next_f32s,
        philox32x4x4_next_f32s,
        ptr,
        out,
        count
    )
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_i32s(
    ptr: *mut Philox32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_rand_i32s,
        philox32x4x4_rand_i32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_f32s(
    ptr: *mut Philox32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_rand_f32s,
        philox32x4x4_rand_f32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_next_u32s_chunk(
    chunk_idx: usize,
    chunk: &mut [u32],
    c: core::arch::x86_64::__m512i,
    k: core::arch::x86_64::__m512i,
    one: core::arch::x86_64::__m512i,
) {
    let offset = (chunk_idx as u128) << PHILOX32x4x4_SHIFT;

    // calculate c
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result = tmp_rng.compute();
            let v = unsafe { _mm512_loadu_si512(result.as_ptr() as *const _) };
            unsafe { _mm512_stream_si512(dst.as_mut_ptr() as *mut _, v) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result = tmp_rng.compute();
            unsafe { copy_nonoverlapping(result.as_ptr(), dst.as_mut_ptr(), PHILOX32x16) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp_rng = Philox32x4x4 { c, k };
        let result = tmp_rng.compute();
        for j in 0..rem.len() {
            rem[j] = result[j];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_next_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c0: core::arch::x86_64::__m512i,
    k: core::arch::x86_64::__m512i,
    one: core::arch::x86_64::__m512i,
    scale: core::arch::x86_64::__m512,
) {
    let offset = chunk_idx as u128 * PHILOX32x4x4_CHUNK_RATIO;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c0) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_mul_ps(v_f32, scale);
            unsafe { _mm512_stream_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_mul_ps(v_f32, scale);
            unsafe { _mm512_storeu_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp_rng = Philox32x4x4 { c, k };
        let result_u32 = tmp_rng.compute();
        let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        let v_res = _mm512_mul_ps(v_f32, scale);
        let mut tmp_f32 = [0f32; 16];
        unsafe { _mm512_storeu_ps(tmp_f32.as_mut_ptr() as *mut _, v_res) };
        for j in 0..rem.len() {
            rem[j] = tmp_f32[j];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_rand_i32s_chunk(
    chunk_idx: usize,
    chunk: &mut [i32],
    c: core::arch::x86_64::__m512i,
    k: core::arch::x86_64::__m512i,
    one: core::arch::x86_64::__m512i,
    v_range: core::arch::x86_64::__m512i,
    v_min: core::arch::x86_64::__m512i,
    merge_mask: u16,
) {
    let offset = (chunk_idx as u128) << PHILOX32x4x4_SHIFT;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0; // N % 64 == 0
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };

            let prod_even = _mm512_mul_epu32(v_u32, v_range);
            let res_even = _mm512_srli_epi64(prod_even, 32);

            let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
            let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

            let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
            let v_res = _mm512_add_epi32(merged, v_min);
            unsafe { _mm512_stream_si512(dst.as_mut_ptr() as *mut _, v_res) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };

            let prod_even = _mm512_mul_epu32(v_u32, v_range);
            let res_even = _mm512_srli_epi64(prod_even, 32);

            let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
            let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

            let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
            let v_res = _mm512_add_epi32(merged, v_min);

            unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, v_res) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp_rng = Philox32x4x4 { c, k };
        let result_u32 = tmp_rng.compute();

        let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);

        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

        let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
        let v_res = _mm512_add_epi32(merged, v_min);

        let mut tmp_res = [0i32; 16];
        unsafe { _mm512_storeu_si512(tmp_res.as_mut_ptr() as *mut _, v_res) };

        for j in 0..rem.len() {
            rem[j] = tmp_res[j];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_rand_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c: core::arch::x86_64::__m512i,
    k: core::arch::x86_64::__m512i,
    one: core::arch::x86_64::__m512i,
    v_mult: core::arch::x86_64::__m512,
    v_min: core::arch::x86_64::__m512,
) {
    let offset = chunk_idx as u128 * PHILOX32x4x4_CHUNK_RATIO;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
            unsafe { _mm512_stream_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let mut tmp_rng = Philox32x4x4 { c, k };
            let result_u32 = tmp_rng.compute();

            let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
            unsafe { _mm512_storeu_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, _mm512_setzero_si512());
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp_rng = Philox32x4x4 { c, k };
        let result_u32 = tmp_rng.compute();
        let v_u32 = unsafe { _mm512_loadu_si512(result_u32.as_ptr() as *const _) };
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
        let mut tmp_f32 = [0f32; 16];
        unsafe { _mm512_storeu_ps(tmp_f32.as_mut_ptr() as *mut _, v_res) };
        for j in 0..rem.len() {
            rem[j] = tmp_f32[j];
        }
    }
}

// --- Xorshift32 ---

/// A 32-bit Xorshift random number generator.
///
/// This generator uses a shift-register based algorithm.
#[repr(C)]
pub struct Xorshift32 {
    a: Wrapping<u32>,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance.
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        Self {
            a: wrap!(sm.nextu()),
        }
    }

    /// Generates the next random `u32` value.
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
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
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

#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_new(seed: u32) -> *mut Xorshift32 {
    Box::into_raw(Box::new(Xorshift32::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_free(ptr: *mut Xorshift32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_next_u32s(ptr: *mut Xorshift32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_next_f32s(ptr: *mut Xorshift32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_rand_i32s(
    ptr: *mut Xorshift32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_rand_f32s(
    ptr: *mut Xorshift32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
/// let mut rng = Xorwow::new(12345);
/// let val = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [Wrapping<u32>; 5],
    c: Wrapping<u32>,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance.
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

    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

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

#[repr(C)]
pub struct SplitMix32 {
    state: Wrapping<u32>,
}

impl SplitMix32 {
    pub fn new(seed: u32) -> Self {
        Self {
            state: wrap!(seed | 1),
        }
    }

    pub fn nextu(&mut self) -> u32 {
        self.state = self.state + wrap!(0x9E3779B9);
        let mut z = self.state;
        z = (z ^ (z >> 16)) + wrap!(0x85ebca6b);
        z = (z ^ (z >> 13)) + wrap!(0xc2b2ae35);
        (z ^ (z >> 16)).0
    }

    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_new(seed: u32) -> *mut SplitMix32 {
    Box::into_raw(Box::new(SplitMix32::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_free(ptr: *mut SplitMix32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_next_u32s(ptr: *mut SplitMix32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_next_f32s(ptr: *mut SplitMix32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_rand_i32s(
    ptr: *mut SplitMix32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randi(min, max);
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_rand_f32s(
    ptr: *mut SplitMix32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
    }
}

// --- Threefry32x4 ---

const THREEFRY32_C240: u32 = 0x1BD11BDA;

/// A Threefry4x32 random number generator (Random123 family).
///
/// This is a counter-based RNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 4 output values per block.
#[repr(C, align(64))]
pub struct Threefry32x4 {
    c: [u32; 4],
    k: [u32; 5],
    tw: [u32; 3],
    index: usize,
    buffer: [u32; 4],
}

impl Threefry32x4 {
    /// Creates a new `Threefry32x4` instance.
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

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_new(seed: u32) -> *mut Threefry32x4 {
    Box::into_raw(Box::new(Threefry32x4::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_free(ptr: *mut Threefry32x4) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
const THREEFRY32_PAR_CHUNK: usize = 4096;

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_u32s(ptr: *mut Threefry32x4, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    dst[2] = result[2];
                    dst[3] = result[3];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_f32s(ptr: *mut Threefry32x4, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = result[0] as f32 * SCALE;
                    dst[1] = result[1] as f32 * SCALE;
                    dst[2] = result[2] as f32 * SCALE;
                    dst[3] = result[3] as f32 * SCALE;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * SCALE;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_i32s(
    ptr: *mut Threefry32x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((result[2] as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((result[3] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_f32s(
    ptr: *mut Threefry32x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    dst[2] = (result[2] as f32 * scale_val) * range_val + min;
                    dst[3] = (result[3] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}

// --- Threefry32x2 ---

/// A Threefry2x32 random number generator.
pub struct Threefry32x2 {
    c: [u32; 2],
    k: [u32; 3],
    buffer: [u32; 2],
    index: usize,
}

impl Threefry32x2 {
    /// Creates a new `Threefry32x2` instance.
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

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_new(seed: u32) -> *mut Threefry32x2 {
    Box::into_raw(Box::new(Threefry32x2::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_free(ptr: *mut Threefry32x2) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const THREEFRY32X2_PAR_CHUNK: usize = 4096;

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_u32s(ptr: *mut Threefry32x2, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_f32s(ptr: *mut Threefry32x2, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = result[0] as f32 * SCALE;
                    dst[1] = result[1] as f32 * SCALE;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * SCALE;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_rand_i32s(
    ptr: *mut Threefry32x2,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_rand_f32s(
    ptr: *mut Threefry32x2,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

// --- Squares32 ---

/// The Squares random number generator (32-bit output version by Bernard Widynski).
pub struct Squares32 {
    c: u64,
    k: u64,
}

impl Squares32 {
    /// Creates a new `Squares32` instance.
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

#[unsafe(no_mangle)]
pub extern "C" fn squares32_new(seed: u64) -> *mut Squares32 {
    Box::into_raw(Box::new(Squares32::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn squares32_free(ptr: *mut Squares32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const SQUARES32_PAR_CHUNK: usize = 4096;

/// 4-way unrolled batch kernel for Squares32.
/// y0..y3 are independent lanes, each advanced by k4 = 4*k per batch.
/// Since z_i = y_i + k, and y_{i+1} = y_i + k, we get z_i == y_{i+1},
/// eliminating redundant adds. No loop-carried dependency within a batch.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_next_u32s(ptr: *mut Squares32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    // z_i = y_i + k == y1 = y0+k, z0 = y1, z1 = y2, z2 = y3
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1);
                    dst[1] = Squares32::compute_yz(y1, y2);
                    dst[2] = Squares32::compute_yz(y2, y3);
                    dst[3] = Squares32::compute_yz(y3, z3);
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr);
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn squares32_next_f32s(ptr: *mut Squares32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1) as f32 * SCALE;
                    dst[1] = Squares32::compute_yz(y1, y2) as f32 * SCALE;
                    dst[2] = Squares32::compute_yz(y2, y3) as f32 * SCALE;
                    dst[3] = Squares32::compute_yz(y3, z3) as f32 * SCALE;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr) as f32 * SCALE;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn squares32_rand_i32s(
    ptr: *mut Squares32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = ((Squares32::compute_yz(y0, y1) as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((Squares32::compute_yz(y1, y2) as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((Squares32::compute_yz(y2, y3) as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((Squares32::compute_yz(y3, z3) as u64 * range) >> 32) as i32 + min;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = ((Squares32::compute_yz(yr, zr) as u64 * range) >> 32) as i32 + min;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn squares32_rand_f32s(
    ptr: *mut Squares32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);
        let combined_scale = (max - min) * (1.0f32 / (u32::MAX as f32 + 1.0));

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1) as f32 * combined_scale + min;
                    dst[1] = Squares32::compute_yz(y1, y2) as f32 * combined_scale + min;
                    dst[2] = Squares32::compute_yz(y2, y3) as f32 * combined_scale + min;
                    dst[3] = Squares32::compute_yz(y3, z3) as f32 * combined_scale + min;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr) as f32 * combined_scale + min;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

#[allow(non_upper_case_globals)]
const SQUARES32x8: usize = 8;

#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct Squares32x8 {
    c: __m512i, // 64x4
    k: __m512i,
}

#[cfg(target_arch = "x86_64")]
impl Squares32x8 {
    /// Creates a new `Squares32x4` instance.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn new(seed: u32) -> Self {
        let mut k = [0u64; SQUARES32x8];
        let mut seedgen = SplitMix64::new(seed as u64 | 1);
        k.iter_mut().for_each(|v| {
            *v = seedgen.nextu();
        });

        unsafe {
            Self {
                c: _mm512_set1_epi64(0),
                k: _mm512_loadu_si512(k.as_ptr() as *const _),
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn compute_yz(y: __m512i, z: __m512i) -> [u32; 8] {
        unsafe {
            // let y = _mm512_loadu_epi64(y.as_ptr() as *const _);
            // let z = _mm512_loadu_epi64(z.as_ptr() as *const _);

            // x = y.wrapping_mul(y).wrapping_add(y);
            let mut x = _mm512_add_epi64(_mm512_mullo_epi64(y, y), y);
            // x = x.rotate_left(32);
            x = _mm512_rol_epi64(x, 32);

            // x = x.wrapping_mul(x).wrapping_add(z);
            x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), z);
            // x = x.rotate_left(32);
            x = _mm512_rol_epi64(x, 32);

            // x = x.wrapping_mul(x).wrapping_add(y);
            x = _mm512_add_epi64(_mm512_mullo_epi64(x, x), y);
            // x = x.rotate_left(32);
            x = _mm512_rol_epi64(x, 32);

            // (x.wrapping_mul(x).wrapping_add(z) >> 32) as u32
            x = _mm512_srli_epi64(_mm512_add_epi64(_mm512_mullo_epi64(x, x), z), 32);

            let dst_v = _mm512_cvtepi64_epi32(x);
            let mut dst = [0u32; SQUARES32x8];
            _mm256_storeu_si256(dst.as_mut_ptr() as *mut _, dst_v);

            dst
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn compute(c: u64, k: u64) -> [u32; SQUARES32x8] {
        unsafe {
            let c = _mm512_add_epi64(
                _mm512_set1_epi64(c as i64),
                _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7), // r: reverse
            );
            let key_vec = _mm512_set1_epi64(k as i64);
            let y_vec = _mm512_mullo_epi64(c, key_vec);
            let z_vec = _mm512_add_epi64(y_vec, key_vec);

            let dst = Self::compute_yz(y_vec, z_vec);
            dst
        }
    }
}

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
            let mut seedgen = SplitMix64::new(1);
            let results = Squares32x8::compute(0, seedgen.nextu());

            assert_eq!(
                results,
                [
                    1225738608, 3944088997, 2344576009, 896386238, 1754778585, 90946642,
                    1788108373, 3630939197
                ]
            );
        }
    }
}
