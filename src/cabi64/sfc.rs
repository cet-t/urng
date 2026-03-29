use crate::rng64::{Sfc64, Sfc64x4, SplitMix64};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Sfc64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`sfc64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_new(seed: u64) -> *mut Sfc64 {
    Box::into_raw(Box::new(Sfc64::new(seed)))
}

/// Frees a `Sfc64` instance previously created by [`sfc64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_free(ptr: *mut Sfc64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Helper: fill a u64 chunk using Sfc64x4 SIMD, falling back to scalar for remainder.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sfc64_fill_u64_avx2(chunk: &mut [u64], seed: u64) {
    unsafe {
        let mut sg = SplitMix64::new(seed);
        let seeds = [sg.nextu(), sg.nextu(), sg.nextu(), sg.nextu()];
        let mut simd = Sfc64x4::new(seeds);
        let len = chunk.len();
        let simd_end = len & !3; // round down to multiple of 4
        let ptr = chunk.as_mut_ptr();
        let mut i = 0;
        while i < simd_end {
            simd.next4u(ptr.add(i));
            i += 4;
        }
        // Scalar fallback for remainder (0-3 elements)
        if i < len {
            let mut scalar = Sfc64::new(seed.wrapping_add(0xDEADBEEF));
            while i < len {
                *ptr.add(i) = scalar.nextu();
                i += 1;
            }
        }
    }
}

/// Helper: fill a f64 chunk using Sfc64x4 SIMD, falling back to scalar for remainder.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sfc64_fill_f64_avx2(chunk: &mut [f64], seed: u64) {
    unsafe {
        let mut sg = SplitMix64::new(seed);
        let seeds = [sg.nextu(), sg.nextu(), sg.nextu(), sg.nextu()];
        let mut simd = Sfc64x4::new(seeds);
        let len = chunk.len();
        let simd_end = len & !3;
        let ptr = chunk.as_mut_ptr();
        let mut i = 0;
        while i < simd_end {
            simd.next4f(ptr.add(i));
            i += 4;
        }
        if i < len {
            let mut scalar = Sfc64::new(seed.wrapping_add(0xDEADBEEF));
            while i < len {
                *ptr.add(i) = scalar.nextf();
                i += 1;
            }
        }
    }
}

/// Helper: fill an i64 chunk using Sfc64x4 SIMD, falling back to scalar for remainder.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sfc64_fill_i64_avx2(chunk: &mut [i64], seed: u64, min: i64, max: i64) {
    unsafe {
        let mut sg = SplitMix64::new(seed);
        let seeds = [sg.nextu(), sg.nextu(), sg.nextu(), sg.nextu()];
        let mut simd = Sfc64x4::new(seeds);
        let len = chunk.len();
        let simd_end = len & !3;
        let ptr = chunk.as_mut_ptr();
        let mut i = 0;
        while i < simd_end {
            simd.next4i(ptr.add(i), min, max);
            i += 4;
        }
        if i < len {
            let mut scalar = Sfc64::new(seed.wrapping_add(0xDEADBEEF));
            while i < len {
                *ptr.add(i) = scalar.randi(min, max);
                i += 1;
            }
        }
    }
}

/// Helper: fill a f64 ranged chunk using Sfc64x4 SIMD, falling back to scalar for remainder.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sfc64_fill_rf64_avx(chunk: &mut [f64], seed: u64, min: f64, max: f64) {
    unsafe {
        let mut sg = SplitMix64::new(seed);
        let seeds = [sg.nextu(), sg.nextu(), sg.nextu(), sg.nextu()];
        let mut simd = Sfc64x4::new(seeds);
        let len = chunk.len();
        let simd_end = len & !3;
        let ptr = chunk.as_mut_ptr();
        let mut i = 0;
        while i < simd_end {
            simd.next4rf(ptr.add(i), min, max);
            i += 4;
        }
        if i < len {
            let mut scalar = Sfc64::new(seed.wrapping_add(0xDEADBEEF));
            while i < len {
                *ptr.add(i) = scalar.randf(min, max);
                i += 1;
            }
        }
    }
}

const SFC64_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values.
/// Uses AVX2 SIMD lanes in parallel chunks on x86_64.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_u64s(ptr: *mut Sfc64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                #[cfg(target_arch = "x86_64")]
                sfc64_fill_u64_avx2(chunk, chunk_seed);
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let mut local_rng = Sfc64::new(chunk_seed);
                    for v in chunk {
                        *v = local_rng.nextu();
                    }
                }
            });
    }
}

/// Fills `out[0..count]` with `f64` values in `[0, 1)`.
/// Uses AVX2 SIMD lanes in parallel chunks on x86_64.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_f64s(ptr: *mut Sfc64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                #[cfg(target_arch = "x86_64")]
                sfc64_fill_f64_avx2(chunk, chunk_seed);
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let mut local_rng = Sfc64::new(chunk_seed);
                    for v in chunk {
                        *v = local_rng.nextf();
                    }
                }
            });
    }
}

/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
/// Uses AVX2 SIMD lanes in parallel chunks on x86_64.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_rand_i64s(
    ptr: *mut Sfc64,
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
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                #[cfg(target_arch = "x86_64")]
                sfc64_fill_i64_avx2(chunk, chunk_seed, min, max);
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let mut local_rng = Sfc64::new(chunk_seed);
                    for v in chunk {
                        *v = local_rng.randi(min, max);
                    }
                }
            });
    }
}

/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
/// Uses AVX2 SIMD lanes in parallel chunks on x86_64.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_rand_f64s(
    ptr: *mut Sfc64,
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
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                #[cfg(target_arch = "x86_64")]
                sfc64_fill_rf64_avx(chunk, chunk_seed, min, max);
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let mut local_rng = Sfc64::new(chunk_seed);
                    for v in chunk {
                        *v = local_rng.randf(min, max);
                    }
                }
            });
    }
}
