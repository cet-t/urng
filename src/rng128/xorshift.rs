use crate::rng::Rng32;
use std::slice::from_raw_parts_mut;

// --- Xorshift128 ---

/// A 128-bit Xorshift random number generator.
///
/// Produces 32-bit output from a 128-bit internal state.
/// Period: 2^128 - 1.
///
/// # Examples
///
/// ```
/// use urng::rng128::Xorshift128;
///
/// let mut rng = Xorshift128::new([1, 2, 3, 4]);
/// assert_eq!(rng.nextu(), 10284);
/// assert_eq!(rng.nextf(), 2.8738286e-6_f32);
/// assert!(rng.randi(1, 100) >= 1);
/// ```
#[repr(C)]
pub struct Xorshift128 {
    x: [u32; 4],
}

impl Xorshift128 {
    /// Creates a new `Xorshift128` instance.
    ///
    /// Each seed element is OR-ed with 1 to prevent an all-zero state.
    pub fn new(seed: [u32; 4]) -> Self {
        Self {
            x: [seed[0] | 1, seed[1] | 1, seed[2] | 1, seed[3] | 1],
        }
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        let mut t = self.x[3];
        t ^= t << 11;
        t ^= t >> 8;
        let s = self.x[0];
        (self.x[1], self.x[2], self.x[3]) = (s, self.x[1], self.x[2]);
        self.x[0] = t ^ s ^ (s >> 19);
        self.x[0]
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

impl Rng32 for Xorshift128 {
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

/// Creates a new heap-allocated `Xorshift128` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xorshift128_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_new(
    seed1: u32,
    seed2: u32,
    seed3: u32,
    seed4: u32,
) -> *mut Xorshift128 {
    Box::into_raw(Box::new(Xorshift128::new([seed1, seed2, seed3, seed4])))
}
/// Frees a `Xorshift128` instance previously created by [`xorshift128_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_free(ptr: *mut Xorshift128) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u32` values from the generator.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_next_u32s(ptr: *mut Xorshift128, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextu();
        }
    }
}
/// Fills `out[0..count]` with `f32` values in `[0, 1)`.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_next_f32s(ptr: *mut Xorshift128, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for v in buffer {
            *v = rng.nextf();
        }
    }
}
/// Fills `out[0..count]` with `i32` values in `[min, max]`.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_rand_i32s(
    ptr: *mut Xorshift128,
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
/// Fills `out[0..count]` with `f32` values in `[min, max)`.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_rand_f32s(
    ptr: *mut Xorshift128,
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
    fn xorshift128_works() {
        let mut rng = Xorshift128::new([1, 2, 3, 4]);
        assert_eq!(rng.nextu(), 10284);
        assert_eq!(rng.nextf(), 2.8738286e-6);
    }
}
