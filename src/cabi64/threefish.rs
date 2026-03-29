use crate::rng64::Threefish256;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Threefish256` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`threefish256_free`].
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_new(seed: u64) -> *mut Threefish256 {
    Box::into_raw(Box::new(Threefish256::new(seed)))
}
/// Frees a `Threefish256` instance previously created by [`threefish256_free`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_free(ptr: *mut Threefish256) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_u64s(ptr: *mut Threefish256, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let out_arr = rng.nextu();
            let limit = (count - i).min(4);
            buffer[i..i + limit].copy_from_slice(&out_arr[..limit]);
            i += 4;
        }
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)`, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_f64s(ptr: *mut Threefish256, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut i = 0;
        while i < count {
            let out_arr = rng.nextf();
            let limit = (count - i).min(4);
            buffer[i..i + limit].copy_from_slice(&out_arr[..limit]);
            i += 4;
        }
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]`, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_rand_i64s(
    ptr: *mut Threefish256,
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
            let out_arr = rng.randi(min, max);
            let limit = (count - i).min(4);
            buffer[i..i + limit].copy_from_slice(&out_arr[..limit]);
            i += 4;
        }
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)`, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_rand_f64s(
    ptr: *mut Threefish256,
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
            let out_arr = rng.randf(min, max);
            let limit = (count - i).min(4);
            buffer[i..i + limit].copy_from_slice(&out_arr[..limit]);
            i += 4;
        }
    }
}
