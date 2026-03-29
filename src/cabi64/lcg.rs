#[allow(deprecated)]
use crate::rng64::Lcg64;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Lcg64` with the given parameters and warm-up count.
/// The caller is responsible for freeing it with [`lcg64_free`].
#[allow(deprecated)]
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_new(x: u64, a: u64, b: u64, m: u64) -> *mut Lcg64 {
    Box::into_raw(Box::new(Lcg64::new(x, a, b, m)))
}

/// Frees a `Lcg64` instance previously created by [`lcg64_new`].
/// Does nothing if `ptr` is null.
#[allow(deprecated)]
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_free(ptr: *mut Lcg64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Fills `out[0..count]` with raw `u64` random values.
#[allow(deprecated)]
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

/// Fills `out[0..count]` with `f64` values uniformly distributed in `[0, 1)`.
#[allow(deprecated)]
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

/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
#[allow(deprecated)]
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

/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
#[allow(deprecated)]
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
