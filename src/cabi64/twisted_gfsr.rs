use crate::rng64::TwistedGFSR;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `TwistedGFSR` using the built-in default seed array.
/// The `_seed` argument is currently unused. The caller must free the result with
/// [`twisted_gfsr_free`].
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_new(_seed: u64) -> *mut TwistedGFSR {
    Box::into_raw(Box::new(TwistedGFSR::new(TwistedGFSR::new_seed())))
}
/// Frees a `TwistedGFSR` instance previously created by [`twisted_gfsr_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_free(ptr: *mut TwistedGFSR) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[0, 1)`.
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
/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
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
