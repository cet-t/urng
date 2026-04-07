use crate::rng::Rng64;
use crate::rng64::{
    SplitMix64,
    xoroshiro::{Xoroshiro128Pp, Xoroshiro128Ss},
};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::slice::from_raw_parts_mut;

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

const XOROSHIRO128PP_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_next_u64s(ptr: *mut Xoroshiro128Pp, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOROSHIRO128PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128pp_next_f64s(ptr: *mut Xoroshiro128Pp, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOROSHIRO128PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
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

        buffer
            .par_chunks_mut(XOROSHIRO128PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
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

        buffer
            .par_chunks_mut(XOROSHIRO128PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
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

const XOROSHIRO128SS_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_next_u64s(ptr: *mut Xoroshiro128Ss, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOROSHIRO128SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro128ss_next_f64s(ptr: *mut Xoroshiro128Ss, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOROSHIRO128SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
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

        buffer
            .par_chunks_mut(XOROSHIRO128SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
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

        buffer
            .par_chunks_mut(XOROSHIRO128SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoroshiro128Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
    }
}
