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
/// Fills `out[0..count]` with raw `u64` random values, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_u64s(ptr: *mut Threefish256, out: *mut u64, count: usize) {
    unimplemented!()
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)`, producing 4 values per cipher block.
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_next_f64s(ptr: *mut Threefish256, out: *mut f64, count: usize) {
    unimplemented!()
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
    unimplemented!()
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
    unimplemented!()
}
