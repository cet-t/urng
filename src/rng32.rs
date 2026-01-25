use crate::{rng::Rng32, wrap};
use std::{hint::black_box, num::Wrapping, slice::from_raw_parts_mut};

// --- Mt19937 ---

/// A 32-bit Mersenne Twister (MT19937) random number generator.
#[repr(C)]
pub struct Mt19937 {
    mt: [u32; N],
    mti: usize,
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908B0DF;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7FFFFFFF;

impl Mt19937 {
    /// Creates a new `Mt19937` instance.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial seed value.
    /// * `warm` - The number of initial iterations to skip (warm-up).
    pub fn new(seed: u32, warm: usize) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = (1812433253u32)
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        let mut rng = Self { mt, mti: N };
        (0..warm).into_iter().for_each(|_| {
            let _ = rng.nextu();
        });
        rng
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
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
pub extern "C" fn mt19937_new(seed: u32, warm: usize) -> *mut Mt19937 {
    Box::into_raw(Box::new(Mt19937::new(seed, warm)))
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

#[repr(C)]
pub struct Lcg32 {
    x: Wrapping<u32>,
    a: u32,
    b: u32,
    m: u32,
    r: f32,
}

impl Lcg32 {
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
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Creates a new `Pcg32` instance.
    pub fn new(seed: u64) -> Self {
        Pcg32 {
            state: seed.wrapping_add(0xDA3E39CB94B95BDB),
            inc: seed | 1,
        }
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let oldstate = self.state;
        self.state = oldstate
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << (rot.wrapping_neg() & 31))
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
    c: [u32; 4],
    k: [u32; 2],
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
    pub fn new(seed: [u32; 2]) -> Self {
        Self {
            c: [1, 0, 0, 0],
            k: seed,
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
        let p0 = self.c[0].wrapping_mul(Self::m0());
        let p1 = self.c[2].wrapping_mul(Self::m1());
        out[0] = p0 ^ self.c[1] ^ self.k[0];
        out[1] = p0;
        out[2] = p1 ^ self.c[3] ^ self.k[1];
        out[3] = p1;
        self.c[0] = self.c[0].wrapping_add(1);
        self.c[1] = self.c[1].wrapping_add(1);
        self.c[2] = self.c[2].wrapping_add(1);
        self.c[3] = self.c[3].wrapping_add(1);
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
pub extern "C" fn philox32_new(seed1: u32, seed2: u32) -> *mut Philox32 {
    Box::into_raw(Box::new(Philox32::new([seed1, seed2])))
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
    a: u32,
}

impl Xorshift32 {
    /// Creates a new `Xorshift32` instance.
    pub fn new(seed: u32) -> Self {
        Self { a: seed | 1 }
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let mut x = self.a;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.a = x;
        x
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

// --- TwistedGFSR ---

/// A Twisted Generalized Feedback Shift Register (TGFSR) generator.
#[repr(C)]
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
    pub fn nextu(&mut self) -> u32 {
        if self.index >= N_GFSR {
            self.twist();
        }
        let mut y = self.seed[self.index];
        y ^= (y << 7) & 0x2b5b2500;
        y ^= (y << 15) & 0xdb8b0000;
        y &= 0xffffffff;
        y ^= y >> 16;
        self.index += 1;
        y as u32
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / 4294967296.0)
    }

    /// Generates a random `i32` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        self.nextf() * range + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for TwistedGFSR {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min as f64, max as f64) as f32
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
/// let mut rng = Xorwow::new(12345);
/// let val = rng.nextu();
/// ```
#[repr(C)]
pub struct Xorwow {
    x: [u32; 5],
    c: u32,
}

impl Xorwow {
    /// Creates a new `Xorwow` instance.
    pub fn new(seed: u32) -> Self {
        let seed = seed | 1;
        Self {
            x: [
                seed,
                seed.rotate_left(8),
                seed.rotate_left(16),
                seed.rotate_left(24),
                seed.rotate_left(32),
            ],
            c: 1,
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
        self.c += 362437;
        t + self.c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_works() {
        let mut rng = Mt19937::new(1, 1024);
        assert_eq!(rng.nextu(), 244660247);
        assert_eq!(rng.nextf(), 0.2754702);
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
        assert_eq!(rng.nextu(), 4164751464);
        assert_eq!(rng.nextf(), 0.9784193);
    }

    #[test]
    fn philox32_works() {
        let mut rng = Philox32::new([1, 2]);
        assert_eq!(rng.nextu(), [3528531794, 3528531795, 2, 0]);
        assert_eq!(rng.nextf(), 0.6431007);
    }

    #[test]
    fn xorshift32_works() {
        let mut rng = Xorshift32::new(1);
        assert_eq!(rng.nextu(), 270369);
        assert_eq!(rng.nextf(), 0.015747428);
    }

    #[test]
    fn twisted_gfsr_works() {
        let mut rng = TwistedGFSR::new(TwistedGFSR::new_seed());
        assert_eq!(rng.nextu(), 868393086);
        assert_eq!(rng.nextf(), 0.33567164628766477);
    }

    #[test]
    fn xorwow_works() {
        let mut rng = Xorwow::new(1);
        assert_eq!(rng.nextu(), 362456);
        // assert_eq!(rng.nextf(), 0.015747428);
    }
}
