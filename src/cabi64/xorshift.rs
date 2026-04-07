use crate::rng::Rng64;
use crate::rng64::Xorshift64;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Xorshift64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xorshift64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_new(seed: u64) -> *mut Xorshift64 {
    Box::into_raw(Box::new(Xorshift64::new(seed)))
}
/// Frees a `Xorshift64` instance previously created by [`xorshift64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_free(ptr: *mut Xorshift64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[0, 1)`.
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
/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
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
