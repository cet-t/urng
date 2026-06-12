use crate::_internal::fill_chunk_auto;
use crate::rng::Rng64;
use crate::rng64::{
    SplitMix64,
    xoroshiro::{Xoroshiro128Pp, Xoroshiro128Ss},
};
use rayon::prelude::*;
use std::slice::from_raw_parts_mut;

const XOROSHIRO128_PAR_CHUNK: usize = 0x20000;

/// Fills `buffer` in parallel: each chunk runs its own decorrelated RNG
/// built by `new_rng`, producing one element per `step` call. Eight
/// outputs (64 bytes for 8-byte `T`) are batched per generator call so
/// the non-temporal path can stream whole cache lines.
#[inline(always)]
fn xoro_fill<R, T, N, M>(buffer: &mut [T], base_seed: u64, new_rng: N, step: M)
where
    T: Copy + Default + Send,
    N: Fn(u64) -> R + Sync,
    M: Fn(&mut R) -> T + Sync,
{
    buffer
        .par_chunks_mut(XOROSHIRO128_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_seed = SplitMix64::compute(
                base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            );
            let mut rng = new_rng(chunk_seed);
            unsafe {
                fill_chunk_auto(chunk, || {
                    let mut out = [T::default(); 8];
                    for v in &mut out {
                        *v = step(&mut rng);
                    }
                    out
                });
            }
        });
}

/// Creates a new heap-allocated `Xoroshiro128Pp` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoroshiro128pp_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_new(seed: u64) -> *mut Xoroshiro128Pp {
    Box::into_raw(Box::new(Xoroshiro128Pp::new(seed)))
}
/// Frees a `Xoroshiro128Pp` instance previously created by [`xoroshiro128pp_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_free(ptr: *mut Xoroshiro128Pp) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_next_u64s(ptr: *mut Xoroshiro128Pp, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Pp::new, |r| r.nextu());
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_next_f64s(ptr: *mut Xoroshiro128Pp, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Pp::new, |r| r.nextf());
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_rand_i64s(
    ptr: *mut Xoroshiro128Pp,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Pp::new, |r| r.randi(min, max));
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_rand_f64s(
    ptr: *mut Xoroshiro128Pp,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Pp::new, |r| r.randf(min, max));
    }
}

/// Creates a new heap-allocated `Xoroshiro128Ss` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoroshiro128ss_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_new(seed: u64) -> *mut Xoroshiro128Ss {
    Box::into_raw(Box::new(Xoroshiro128Ss::new(seed)))
}
/// Frees a `Xoroshiro128Ss` instance previously created by [`xoroshiro128ss_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_free(ptr: *mut Xoroshiro128Ss) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_next_u64s(ptr: *mut Xoroshiro128Ss, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Ss::new, |r| r.nextu());
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_next_f64s(ptr: *mut Xoroshiro128Ss, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Ss::new, |r| r.nextf());
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_rand_i64s(
    ptr: *mut Xoroshiro128Ss,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Ss::new, |r| r.randi(min, max));
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_rand_f64s(
    ptr: *mut Xoroshiro128Ss,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();
        xoro_fill(buffer, base_seed, Xoroshiro128Ss::new, |r| r.randf(min, max));
    }
}
