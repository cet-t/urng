#[allow(deprecated)]
use crate::rng32::Lcg32;
use std::slice::from_raw_parts_mut;

/// Creates a new `Lcg32` instance on the heap.
/// The caller is responsible for freeing the memory using `lcg32_free`.
#[allow(deprecated)]
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_new(x: u32, a: u32, b: u32, m: u32) -> *mut Lcg32 {
    Box::into_raw(Box::new(Lcg32::new(x, a, b, m)))
}
/// Frees the memory of a `Lcg32` instance.
#[allow(deprecated)]
#[unsafe(no_mangle)]
pub extern "C" fn lcg32_free(ptr: *mut Lcg32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills the output buffer with the next random `u32` values.
#[allow(deprecated)]
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
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[allow(deprecated)]
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
/// Fills the output buffer with random `i32` values in the range [min, max].
#[allow(deprecated)]
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
/// Fills the output buffer with random `f32` values in the range [min, max).
#[allow(deprecated)]
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
