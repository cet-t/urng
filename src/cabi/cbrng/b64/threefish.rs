#![allow(unused)]

use std::slice::from_raw_parts_mut;

use crate::{cbrng::b64::Threefish256, i2f_bits, u2f_01};

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
/// Fills `buffer` sequentially, mapping each raw `u64` through `map`.
/// Each cipher block yields 4 values; the tail uses a partial block.
#[inline(always)]
fn threefish256_fill<T, M>(rng: &mut Threefish256, buffer: &mut [T], map: M)
where
    M: Fn(u64) -> T,
{
    for chunk in buffer.chunks_mut(4) {
        let r = rng.next_raw();
        for (dst, v) in chunk.iter_mut().zip(r) {
            *dst = map(*v);
        }
    }
}

/// Fills `out[0..count]` with raw `u64` random values, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_u64s(ptr: *mut Threefish256, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        threefish256_fill(rng, buffer, |x| x);
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)`, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_f64s(ptr: *mut Threefish256, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        threefish256_fill(rng, buffer, |x| u2f_01!(f64, 64, x));
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
        let range = (max as i128 - min as i128 + 1) as u128;
        threefish256_fill(rng, buffer, |x| ((x as u128 * range) >> 64) as i64 + min);
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
        let mult = max - min;
        threefish256_fill(rng, buffer, |x| u2f_01!(f64, 64, x) * mult + min);
    }
}
