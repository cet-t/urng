use crate::rng::Rng64;
use crate::wrap;
use std::hint::black_box;
use std::num::Wrapping;
use std::slice::from_raw_parts_mut;

// --- Mt1993764 ---

/// A 64-bit Mersenne Twister (MT19937-64) random number generator.
#[repr(C)]
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
    /// Creates a new `Mt1993764` instance.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial seed value.
    /// * `warm` - The number of initial iterations to skip (warm-up).
    pub fn new(seed: u64, warm: usize) -> Self {
        let mut mt = [0u64; N];
        let mut seedgen = SplitMix64::new(seed);
        mt[0] = seedgen.nextu();
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = 6364136223846793005u64
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        let mut rng = Self { mt, mti: N };
        (0..warm).into_iter().for_each(|_| {
            let _ = rng.nextu();
        });
        rng
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

#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_new(seed: u64, warm: usize) -> *mut Mt1993764 {
    Box::into_raw(Box::new(Mt1993764::new(seed, warm)))
}
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_free(ptr: *mut Mt1993764) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_next_u64s(ptr: *mut Mt1993764, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_next_f64s(ptr: *mut Mt1993764, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_rand_i64s(
    ptr: *mut Mt1993764,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn mt1993764_rand_f64s(
    ptr: *mut Mt1993764,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
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

#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_new(_seed: u64) -> *mut TwistedGFSR {
    Box::into_raw(Box::new(TwistedGFSR::new(TwistedGFSR::new_seed())))
}
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_free(ptr: *mut TwistedGFSR) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_next_u64s(ptr: *mut TwistedGFSR, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_next_f64s(ptr: *mut TwistedGFSR, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_rand_i64s(
    ptr: *mut TwistedGFSR,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn twisted_gfsr_rand_f64s(
    ptr: *mut TwistedGFSR,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
    }
}

// --- Lcg64 ---

/// A Linear Congruential Generator (LCG) for 64-bit random numbers.
///
/// This generator produces pseudo-random numbers using the recurrence relation:
/// X(n+1) = (a * X(n) + b) % M
#[repr(C)]
pub struct Lcg64 {
    x: Wrapping<u64>,
    a: u64,
    b: u64,
    m: u64,
    r: f64,
}

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

    #[inline]
    pub fn nextu(&mut self) -> u64 {
        self.x = wrap!((self.a * self.x.0 + self.b) % self.m);
        self.x.0
    }

    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * self.r
    }

    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

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

#[unsafe(no_mangle)]
pub extern "C" fn lcg64_new(x: u64, a: u64, b: u64, m: u64, warm: usize) -> *mut Lcg64 {
    Box::into_raw(Box::new(Lcg64::new(x, a, b, m, warm)))
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_free(ptr: *mut Lcg64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_next_u64s(ptr: *mut Lcg64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_next_f64s(ptr: *mut Lcg64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_rand_i64s(
    ptr: *mut Lcg64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn lcg64_rand_f64s(
    ptr: *mut Lcg64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
    }
}

// --- Philox64 ---

/// A Philox 2x64 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
#[repr(C)]
pub struct Philox64 {
    c: [u64; 2],
    k: [u64; 2],
}

impl Philox64 {
    const fn chunk_size() -> usize {
        2
    }
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

    /// Generates the next block of random numbers.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 2] {
        let mut v0 = self.c[0];
        let mut v1 = self.c[1];
        let mut k = self.k[0];

        let w0: u64 = 0x9E3779B97F4A7C15;

        for _ in 0..10 {
            let prod = (v0 as u128).wrapping_mul(Self::m0());
            let hi = (prod >> 64) as u64;
            let lo = prod as u64;
            let next_v0 = hi ^ v1 ^ k;
            let next_v1 = lo;

            v0 = next_v0;
            v1 = next_v1;
            k = k.wrapping_add(w0);
        }

        self.c[0] = self.c[0].wrapping_add(1);
        if self.c[0] == 0 {
            self.c[1] = self.c[1].wrapping_add(1);
        }

        [v0, v1]
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

#[unsafe(no_mangle)]
pub extern "C" fn philox64_new(seed: u64) -> *mut Philox64 {
    Box::into_raw(Box::new(Philox64::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn philox64_free(ptr: *mut Philox64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_u64s(ptr: *mut Philox64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox64::chunk_size());
            buffer[i..i + take].copy_from_slice(&chunk[..take]);
            i += take;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_f64s(ptr: *mut Philox64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox64::chunk_size());
            for j in 0..take {
                buffer[i + j] = chunk[j] as f64 * (1.0 / (u64::MAX as f64 + 1.0));
            }
            i += take;
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_i64s(
    ptr: *mut Philox64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox64::chunk_size());
            for j in 0..take {
                let range = (max as i128 - min as i128 + 1) as u128;
                buffer[i + j] = ((chunk[j] as u128 * range) >> 64) as i64 + min;
            }
            i += take;
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_f64s(
    ptr: *mut Philox64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let chunk = rng.nextu();
            let take = (count - i).min(Philox64::chunk_size());
            for j in 0..take {
                let val_01 = chunk[j] as f64 * (1.0 / (u64::MAX as f64 + 1.0));
                buffer[i + j] = val_01 * (max - min) + min;
            }
            i += take;
        }
    }
}

// --- Sfc64 ---

#[repr(C)]
/// A 64-bit SFC random number generator.
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
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let res = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = res.rotate_left(24);

        self.counter = self.counter.wrapping_add(1);
        res
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

impl Rng64 for Sfc64 {
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

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_new(seed: u64) -> *mut Sfc64 {
    Box::into_raw(Box::new(Sfc64::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_free(ptr: *mut Sfc64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_u64s(ptr: *mut Sfc64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_f64s(ptr: *mut Sfc64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_rand_i64s(
    ptr: *mut Sfc64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn sfc64_rand_f64s(
    ptr: *mut Sfc64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
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

#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_new(seed: u64) -> *mut Xorshift64 {
    Box::into_raw(Box::new(Xorshift64::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_free(ptr: *mut Xorshift64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_next_u64s(ptr: *mut Xorshift64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_next_f64s(ptr: *mut Xorshift64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_rand_i64s(
    ptr: *mut Xorshift64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn xorshift64_rand_f64s(
    ptr: *mut Xorshift64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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

    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn cet64_new(seed: u64) -> *mut Cet64 {
    Box::into_raw(Box::new(Cet64::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn cet64_free(ptr: *mut Cet64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn cet64_next_u64s(ptr: *mut Cet64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn cet64_next_f64s(ptr: *mut Cet64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn cet64_rand_i64s(
    ptr: *mut Cet64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn cet64_rand_f64s(
    ptr: *mut Cet64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
/// let mut rng = Xoshiro256Pp::new(12345);
/// let val = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoshiro256Pp {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Pp {
    /// Creates a new `Xoshiro256Pp` instance.
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

    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_new(seed: u64) -> *mut Xoshiro256Pp {
    Box::into_raw(Box::new(Xoshiro256Pp::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_free(ptr: *mut Xoshiro256Pp) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_u64s(ptr: *mut Xoshiro256Pp, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_f64s(ptr: *mut Xoshiro256Pp, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_rand_i64s(
    ptr: *mut Xoshiro256Pp,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn xoshiro256pp_rand_f64s(
    ptr: *mut Xoshiro256Pp,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
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
/// let mut rng = Xoshiro256Ss::new(12345);
/// let val = rng.nextu();
/// ```
#[repr(C)]
pub struct Xoshiro256Ss {
    s: [Wrapping<u64>; 4],
}

impl Xoshiro256Ss {
    /// Creates a new `Xoshiro256Ss` instance.
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

    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_new(seed: u64) -> *mut Xoshiro256Ss {
    Box::into_raw(Box::new(Xoshiro256Ss::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_free(ptr: *mut Xoshiro256Ss) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_u64s(ptr: *mut Xoshiro256Ss, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_f64s(ptr: *mut Xoshiro256Ss, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_rand_i64s(
    ptr: *mut Xoshiro256Ss,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn xoshiro256ss_rand_f64s(
    ptr: *mut Xoshiro256Ss,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.randf(min, max);
        }
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
/// let mut rng = SplitMix64::new(12345);
/// let val = rng.nextu();
/// ```
#[repr(C)]
pub struct SplitMix64 {
    s: Wrapping<u64>,
}

impl SplitMix64 {
    /// Creates a new `SplitMix64` instance.
    pub fn new(seed: u64) -> Self {
        Self { s: wrap!(seed | 1) }
    }

    /// Generates the next random `u64` value.
    pub fn nextu(&mut self) -> u64 {
        self.s += 0x9E3779B97F4A7C15;
        let mut result = self.s;
        result = (result ^ (result >> 30)) * wrap!(0xBF58476D1CE4E5B9);
        result = (result ^ (result >> 27)) * wrap!(0x94D049BB133111EB);
        (result ^ (result >> 31)).0
    }

    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
    }

    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 64) as i64 + min
    }

    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

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

#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_new(seed: u64) -> *mut SplitMix64 {
    Box::into_raw(Box::new(SplitMix64::new(seed)))
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_free(ptr: *mut SplitMix64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_u64s(ptr: *mut SplitMix64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_f64s(ptr: *mut SplitMix64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_rand_i64s(
    ptr: *mut SplitMix64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
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
pub extern "C" fn splitmix64_rand_f64s(
    ptr: *mut SplitMix64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
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
    fn mt1993764_works() {
        let mut rng = Mt1993764::new(1, 1024);
        assert_eq!(rng.nextu(), 17135235817683363880);
        assert_eq!(rng.nextf(), 0.5149867566929178);
    }

    #[test]
    fn twisted_gfsr_works() {
        let mut rng = TwistedGFSR::new(TwistedGFSR::new_seed());
        assert_eq!(rng.nextu(), 868393086);
        assert_eq!(rng.nextf(), 0.33567164628766477);
    }

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
}
