use crate::rng::Rng32;
use crate::rng32::{Xorshift32, Xorshift128};
use std::slice::from_raw_parts_mut;

/// Creates a new `Xorshift32` instance.
/// The caller is responsible for freeing the memory using `xorshift32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_new(seed: u32) -> *mut Xorshift32 {
    Box::into_raw(Box::new(Xorshift32::new(seed)))
}

/// Frees the memory of a `Xorshift32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_free(ptr: *mut Xorshift32) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_next_u32s(ptr: *mut Xorshift32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextu();
        }
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn xorshift32_next_f32s(ptr: *mut Xorshift32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
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
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
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
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

/// Creates a new heap-allocated `Xorshift128` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xorshift128_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xorshift128_new(seed: u32) -> *mut Xorshift128 {
    Box::into_raw(Box::new(Xorshift128::new(seed)))
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
