use crate::rng::Rng64;
use crate::wrap;
use std::f64::consts::{E, PI};
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
        mt[0] = seed;
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
    pub fn new(seed: [u64; 2]) -> Self {
        Self { c: [1, 0], k: seed }
    }

    /// Advances the generator state by `count` steps.
    pub fn warm(&mut self, count: usize) {
        for _i in 0..count {
            let _ = self.nextu();
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
        self.nextf() * (max - min) + min
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
pub extern "C" fn philox64_new(seed1: u64, seed2: u64) -> *mut Philox64 {
    Box::into_raw(Box::new(Philox64::new([seed1, seed2])))
}
#[unsafe(no_mangle)]
pub extern "C" fn philox64_warm(ptr: *mut Philox64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        rng.warm(count);
    }
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
        Self {
            a: seed,
            b: seed.wrapping_add(E.to_bits()),
            c: seed.wrapping_add(PI.to_bits()),
            counter: 0,
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
        Self { a: seed }
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

#[repr(C)]
pub struct Cet64 {
    k: Wrapping<u64>,
    v: [Wrapping<u64>; 4],
    c: Wrapping<u64>,
}

impl Cet64 {
    pub fn new(seed: u64) -> Self {
        let k = seed | 1;
        Self {
            k: wrap!(k),
            v: [
                wrap!(seed),
                wrap!(k),
                wrap!(k.reverse_bits() | 1),
                wrap!(seed.rotate_left(64)),
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

#[repr(C)]
pub struct Cet364 {
    k: Wrapping<u64>,
    v: [Wrapping<u64>; 3],
    c: Wrapping<u64>,
}

impl Cet364 {
    pub fn new(seed: u64) -> Self {
        let k = seed | 1;
        Self {
            k: wrap!(k),
            v: [wrap!(seed), wrap!(k), wrap!(k.reverse_bits() | 1)],
            c: wrap!(1327),
        }
    }

    #[inline]
    pub fn nextu(&mut self) -> u64 {
        let [mut a, mut b, mut c] = self.v;

        a = (a + self.c) * self.k;

        b ^= c.0.rotate_left(32);
        c += 1327;

        self.c += 1327;
        self.v = [a, b, c];

        a.0 ^ b.0 ^ c.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt1993764_works() {
        let mut rng = Mt1993764::new(1, 1024);
        assert_eq!(rng.nextu(), 4804558718269354394);
        assert_eq!(rng.nextf(), 0.6894973013138909);
    }

    #[test]
    fn philox64_works() {
        let mut rng = Philox64::new([1, 2]);
        assert_eq!(rng.nextu(), [15183679468541472402, 0]);
        assert_eq!(rng.nextf(), 0.6462178266116214);
    }

    #[test]
    fn xorshift64_works() {
        let mut rng = Xorshift64::new(1);
        assert_eq!(rng.nextu(), 1082269761);
        assert_eq!(rng.nextf(), 0.06250387570981203);
    }

    #[test]
    fn sfc64_works() {
        let mut rng = Sfc64::new(1);
        let val1 = rng.nextu();
        let val2 = rng.nextf();
        assert_eq!(val1, 4613303445314885483);
        assert_eq!(val2, 0.5014639839943512);
    }

    #[test]
    fn cet64_works() {
        let mut rng = Cet64::new(1);
        let val1 = rng.nextu();
        assert_eq!(val1, 5981625010123571202);
    }
}
