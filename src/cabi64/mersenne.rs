use crate::rng64::{Mt1993764, Sfmt1993764};
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Mt1993764` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`mt1993764_free`].
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_new(seed: u64) -> *mut Mt1993764 {
    Box::into_raw(Box::new(Mt1993764::new(seed)))
}
/// Frees a `Mt1993764` instance previously created by [`mt1993764_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn mt1993764_free(ptr: *mut Mt1993764) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[0, 1)`.
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
/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
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
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
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

/// Creates a new heap-allocated `Sfmt1993764` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`sfmt1993764_free`].
#[unsafe(no_mangle)]
pub extern "C" fn sfmt1993764_new(seed: u64) -> *mut Sfmt1993764 {
    Box::into_raw(Box::new(Sfmt1993764::new(seed)))
}
/// Frees a `Sfmt1993764` instance previously created by [`sfmt1993764_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn sfmt1993764_free(ptr: *mut Sfmt1993764) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
#[unsafe(no_mangle)]
pub extern "C" fn sfmt1993764_next_u64s(ptr: *mut Sfmt1993764, out: *mut u64, count: usize) {
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
pub extern "C" fn sfmt1993764_next_f64s(ptr: *mut Sfmt1993764, out: *mut f64, count: usize) {
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
pub extern "C" fn sfmt1993764_rand_i64s(
    ptr: *mut Sfmt1993764,
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
pub extern "C" fn sfmt_rand_f64s(
    ptr: *mut Sfmt1993764,
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
