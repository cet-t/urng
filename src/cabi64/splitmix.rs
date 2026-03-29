use crate::rng64::SplitMix64;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `SplitMix64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`splitmix64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_new(seed: u64) -> *mut SplitMix64 {
    Box::into_raw(Box::new(SplitMix64::new(seed)))
}
/// Frees a `SplitMix64` instance previously created by [`splitmix64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_free(ptr: *mut SplitMix64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const SPLITMIX64_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_u64s(ptr: *mut SplitMix64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let s0 = rng.s.0;

        buffer
            .par_chunks_mut(SPLITMIX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut start_idx = (chunk_idx * SPLITMIX64_PAR_CHUNK) as u64;
                for v in chunk {
                    start_idx += 1;
                    let state = s0.wrapping_add(start_idx.wrapping_mul(0x9E3779B97F4A7C15));
                    *v = SplitMix64::compute(state);
                }
            });

        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15));
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_f64s(ptr: *mut SplitMix64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let s0 = rng.s.0;
        let scale = 1.0f64 / (u64::MAX as f64 + 1.0);

        buffer
            .par_chunks_mut(SPLITMIX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut start_idx = (chunk_idx * SPLITMIX64_PAR_CHUNK) as u64;
                for v in chunk {
                    start_idx += 1;
                    let state = s0.wrapping_add(start_idx.wrapping_mul(0x9E3779B97F4A7C15));
                    *v = SplitMix64::compute(state) as f64 * scale;
                }
            });

        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15));
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_rand_i64s(
    ptr: *mut SplitMix64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let s0 = rng.s.0;
        let range = (max as i128 - min as i128 + 1) as u128;

        buffer
            .par_chunks_mut(SPLITMIX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut start_idx = (chunk_idx * SPLITMIX64_PAR_CHUNK) as u64;
                for v in chunk {
                    start_idx += 1;
                    let state = s0.wrapping_add(start_idx.wrapping_mul(0x9E3779B97F4A7C15));
                    let val = SplitMix64::compute(state);
                    *v = ((val as u128 * range) >> 64) as i64 + min;
                }
            });

        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15));
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_rand_f64s(
    ptr: *mut SplitMix64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let s0 = rng.s.0;
        let scale_val = 1.0f64 / (u64::MAX as f64 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(SPLITMIX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut start_idx = (chunk_idx * SPLITMIX64_PAR_CHUNK) as u64;
                for v in chunk {
                    start_idx += 1;
                    let state = s0.wrapping_add(start_idx.wrapping_mul(0x9E3779B97F4A7C15));
                    let val_01 = SplitMix64::compute(state) as f64 * scale_val;
                    *v = val_01 * range_val + min;
                }
            });

        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15));
    }
}
