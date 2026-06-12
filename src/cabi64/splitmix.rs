use crate::_internal::fill_chunk_auto;
use crate::rng64::SplitMix64;
use rayon::prelude::*;
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

const SPLITMIX64_PAR_CHUNK: usize = 0x20000;
const SPLITMIX64_GAMMA: u64 = 0x9E3779B97F4A7C15;

/// Fills `buffer` in parallel from counter-mode SplitMix64 states
/// (`s0 + index * GAMMA`), mapping each raw `u64` through `map`. Eight
/// outputs (64 bytes for 8-byte `T`) are batched per generator call so
/// the non-temporal path can stream whole cache lines.
#[inline(always)]
fn sm64_fill<T, M>(buffer: &mut [T], s0: u64, map: M)
where
    T: Copy + Default + Send,
    M: Fn(u64) -> T + Sync,
{
    buffer
        .par_chunks_mut(SPLITMIX64_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut idx = (chunk_idx * SPLITMIX64_PAR_CHUNK) as u64;
            unsafe {
                fill_chunk_auto(chunk, || {
                    let mut out = [T::default(); 8];
                    for v in &mut out {
                        idx += 1;
                        let state = s0.wrapping_add(idx.wrapping_mul(SPLITMIX64_GAMMA));
                        *v = map(SplitMix64::compute(state));
                    }
                    out
                });
            }
        });
}

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_u64s(ptr: *mut SplitMix64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        sm64_fill(buffer, rng.s.0, |x| x);
        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(SPLITMIX64_GAMMA));
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix64_next_f64s(ptr: *mut SplitMix64, out: *mut f64, count: usize) {
    const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        sm64_fill(buffer, rng.s.0, |x| x as f64 * SCALE);
        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(SPLITMIX64_GAMMA));
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
        let range = (max as i128 - min as i128 + 1) as u128;
        sm64_fill(buffer, rng.s.0, |x| ((x as u128 * range) >> 64) as i64 + min);
        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(SPLITMIX64_GAMMA));
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
    const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mult = (max - min) * SCALE;
        sm64_fill(buffer, rng.s.0, |x| x as f64 * mult + min);
        rng.s.0 = rng
            .s
            .0
            .wrapping_add((count as u64).wrapping_mul(SPLITMIX64_GAMMA));
    }
}
