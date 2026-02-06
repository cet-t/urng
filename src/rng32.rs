use crate::{rng::Rng32, rng64::SplitMix64, wrap};
use std::{hint::black_box, num::Wrapping, slice::from_raw_parts_mut};

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

/// A Philox 4x32 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[repr(C)]
pub struct Philox32 {
    c: [Wrapping<u32>; 4],
    k: [Wrapping<u32>; 2],
}

impl Philox32 {
    const fn chunk_size() -> usize {
        4
    }
    const fn m0() -> u32 {
        0xD2511F53
    }
    const fn m1() -> u32 {
        0xCD9E8D57
    }

    /// Creates a new `Philox32` instance.
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        Self {
            c: [
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
                wrap!(seedgen.nextu()),
            ],
            k: [wrap!(seedgen.nextu()), wrap!(seedgen.nextu())],
        }
    }

    /// Advances the generator state by `count` steps.
    pub fn warm(&mut self, count: usize) {
        for _i in 0..count {
            let _ = self.nextu();
        }
    }

    /// Generates the next block of random numbers.
    #[inline]
    pub fn nextu(&mut self) -> [u32; 4] {
        let mut out = [0u32; 4];
        let p0 = self.c[0] * wrap!(Self::m0());
        let p1 = self.c[2] * wrap!(Self::m1());
        out[0] = p0.0 ^ self.c[1].0 ^ self.k[0].0;
        out[1] = p0.0;
        out[2] = p1.0 ^ self.c[3].0 ^ self.k[1].0;
        out[3] = p1.0;
        self.c[0] += 1;
        self.c[1] += 1;
        self.c[2] += 1;
        self.c[3] += 1;
        out
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu()[0] as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu()[0];
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu()[0] as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Philox32 {
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
pub extern "C" fn philox32_new(seed: u32) -> *mut Philox32 {
    Box::into_raw(Box::new(Philox32::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_warm(ptr: *mut Philox32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        rng.warm(count);
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_free(ptr: *mut Philox32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_u32s(ptr: *mut Philox32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox32::chunk_size());
            buffer[i..i + take].copy_from_slice(&chunk[..take]);
            i += take;
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_f32s(ptr: *mut Philox32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox32::chunk_size());
            for j in 0..take {
                buffer[i + j] = chunk[j] as f32 * (1.0 / (u32::MAX as f32 + 1.0));
            }
            i += take;
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_i32s(
    ptr: *mut Philox32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox32::chunk_size());
            for j in 0..take {
                let range = (max as i64 - min as i64 + 1) as u64;
                buffer[i + j] = ((chunk[j] as u64 * range) >> 32) as i32 + min;
            }
            i += take;
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_f32s(
    ptr: *mut Philox32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox32::chunk_size());
            for j in 0..take {
                let scale = (max - min) * (1.0 / (u32::MAX as f32 + 1.0));
                buffer[i + j] = (chunk[j] as f32 * scale) + min;
            }
            i += take;
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
        let mut rng = Philox32::new(1);
        assert_eq!(rng.nextu(), [2944603146, 1966386973, 963419730, 83976542]);
        assert_eq!(rng.nextf(), 0.61405116);
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
}
