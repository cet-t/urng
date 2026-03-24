use crate::rng64::*;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::arch::x86_64::_mm512_storeu_si512;
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

/// Creates a new heap-allocated `TwistedGFSR` using the built-in default seed array.
/// The `_seed` argument is currently unused. The caller must free the result with
/// [`twisted_gfsr_free`].
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_new(_seed: u64) -> *mut TwistedGFSR {
    Box::into_raw(Box::new(TwistedGFSR::new(TwistedGFSR::new_seed())))
}
/// Frees a `TwistedGFSR` instance previously created by [`twisted_gfsr_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_free(ptr: *mut TwistedGFSR) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
#[unsafe(no_mangle)]
pub extern "C" fn twisted_gfsr_next_u64s(ptr: *mut TwistedGFSR, out: *mut u64, count: usize) {
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
pub extern "C" fn twisted_gfsr_next_f64s(ptr: *mut TwistedGFSR, out: *mut f64, count: usize) {
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
pub extern "C" fn twisted_gfsr_rand_i64s(
    ptr: *mut TwistedGFSR,
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
pub extern "C" fn twisted_gfsr_rand_f64s(
    ptr: *mut TwistedGFSR,
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

/// Creates a new heap-allocated `Lcg64` with the given parameters and warm-up count.
/// The caller is responsible for freeing it with [`lcg64_free`].
#[allow(deprecated)]
#[unsafe(no_mangle)]
pub extern "C" fn lcg64_new(x: u64, a: u64, b: u64, m: u64, warm: usize) -> *mut Lcg64 {
    Box::into_raw(Box::new(Lcg64::new(x, a, b, m, warm)))
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

/// Creates a new heap-allocated `Philox64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`philox64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn philox64_new(seed: u64) -> *mut Philox64 {
    Box::into_raw(Box::new(Philox64::new(seed)))
}
/// Frees a `Philox64` instance previously created by [`philox64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_free(ptr: *mut Philox64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) }
    }
}

/// Parallel batch size for Philox64 (elements per thread task).
const PHILOX64_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_u64s(ptr: *mut Philox64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;

        buffer
            .par_chunks_mut(PHILOX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX64_PAR_CHUNK) / 2;
                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = ((count + 1) / 2) as u64;
        let (new_c0, carry) = rng.c[0].overflowing_add(num_blocks);
        rng.c[0] = new_c0;
        if carry {
            rng.c[1] = rng.c[1].wrapping_add(1);
        }
    }
}

/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_f64s(ptr: *mut Philox64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale = 1.0f64 / (u64::MAX as f64 + 1.0);

        buffer
            .par_chunks_mut(PHILOX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX64_PAR_CHUNK) / 2;
                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    dst[0] = result[0] as f64 * scale;
                    dst[1] = result[1] as f64 * scale;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f64 * scale;
                    }
                }
            });

        let num_blocks = ((count + 1) / 2) as u64;
        let (new_c0, carry) = rng.c[0].overflowing_add(num_blocks);
        rng.c[0] = new_c0;
        if carry {
            rng.c[1] = rng.c[1].wrapping_add(1);
        }
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_i64s(
    ptr: *mut Philox64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let range = (max as i128 - min as i128 + 1) as u128;

        buffer
            .par_chunks_mut(PHILOX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX64_PAR_CHUNK) / 2;
                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    dst[0] = ((result[0] as u128 * range) >> 64) as i64 + min;
                    dst[1] = ((result[1] as u128 * range) >> 64) as i64 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u128 * range) >> 64) as i64 + min;
                    }
                }
            });

        let num_blocks = ((count + 1) / 2) as u64;
        let (new_c0, carry) = rng.c[0].overflowing_add(num_blocks);
        rng.c[0] = new_c0;
        if carry {
            rng.c[1] = rng.c[1].wrapping_add(1);
        }
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_f64s(
    ptr: *mut Philox64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale_val = 1.0f64 / (u64::MAX as f64 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(PHILOX64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX64_PAR_CHUNK) / 2;
                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    dst[0] = (result[0] as f64 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f64 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].overflowing_add((chunk_base_block as u64) + b_offset);
                    c[0] = new_c0;
                    if overflow {
                        c[1] = c[1].wrapping_add(1);
                    }

                    let result = Philox64::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f64 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = ((count + 1) / 2) as u64;
        let (new_c0, carry) = rng.c[0].overflowing_add(num_blocks);
        rng.c[0] = new_c0;
        if carry {
            rng.c[1] = rng.c[1].wrapping_add(1);
        }
    }
}

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

/// Creates a new heap-allocated `Xorshift64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xorshift64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_new(seed: u64) -> *mut Xorshift64 {
    Box::into_raw(Box::new(Xorshift64::new(seed)))
}
/// Frees a `Xorshift64` instance previously created by [`xorshift64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_free(ptr: *mut Xorshift64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
#[unsafe(no_mangle)]
pub extern "C" fn xorshift64_next_u64s(ptr: *mut Xorshift64, out: *mut u64, count: usize) {
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
pub extern "C" fn xorshift64_next_f64s(ptr: *mut Xorshift64, out: *mut f64, count: usize) {
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
pub extern "C" fn xorshift64_rand_i64s(
    ptr: *mut Xorshift64,
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
pub extern "C" fn xorshift64_rand_f64s(
    ptr: *mut Xorshift64,
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

/// Creates a new heap-allocated `Cet64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`cet64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn cet64_new(seed: u64) -> *mut Cet64 {
    Box::into_raw(Box::new(Cet64::new(seed)))
}
/// Frees a `Cet64` instance previously created by [`cet64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_free(ptr: *mut Cet64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_next_u64s(ptr: *mut Cet64, out: *mut u64, count: usize) {
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
pub extern "C" fn cet64_next_f64s(ptr: *mut Cet64, out: *mut f64, count: usize) {
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
pub extern "C" fn cet64_rand_i64s(
    ptr: *mut Cet64,
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
pub extern "C" fn cet64_rand_f64s(
    ptr: *mut Cet64,
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

/// Creates a new heap-allocated `Xoshiro256Pp` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256pp_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_new(seed: u64) -> *mut Xoshiro256Pp {
    Box::into_raw(Box::new(Xoshiro256Pp::new(seed)))
}
/// Frees a `Xoshiro256Pp` instance previously created by [`xoshiro256pp_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_free(ptr: *mut Xoshiro256Pp) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256PP_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_u64s(ptr: *mut Xoshiro256Pp, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_f64s(ptr: *mut Xoshiro256Pp, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_rand_i64s(
    ptr: *mut Xoshiro256Pp,
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
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_rand_f64s(
    ptr: *mut Xoshiro256Pp,
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
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
    }
}

/// Creates a new heap-allocated `Xoshiro256Ssx2` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256ssx2_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_new(seed: u64) -> *mut Xoshiro256Ssx2 {
    Box::into_raw(Box::new(Xoshiro256Ssx2::new(seed)))
}
/// Frees a `Xoshiro256Ssx2` instance previously created by [`xoshiro256ssx2_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_free(ptr: *mut Xoshiro256Ssx2) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256SSX2_PAR_CHUNK: usize = 131_072;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn xoshiro256ssx2_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], base_seed: u64) {
    // 4-way interleaved xoshiro256++ with AVX-512 SoA layout.
    // Each group holds 8 independent xoshiro256++ streams (SoA: one __m512i per state word).
    // 4 groups × 8 streams = 32 independent streams total, producing 32 u64 per iteration.
    let stride = 0x9E3779B97F4A7C15u64;
    let chunk_base = base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(stride));

    macro_rules! make_state_vec {
        ($group:expr) => {{
            let mut sv = [[0u64; 8]; 4];
            for i in 0usize..8 {
                let seed = SplitMix64::compute(
                    chunk_base.wrapping_add((($group * 8 + i) as u64).wrapping_mul(stride)),
                );
                let mut sm = SplitMix64::new(seed);
                sv[0][i] = sm.nextu();
                sv[1][i] = sm.nextu();
                sv[2][i] = sm.nextu();
                sv[3][i] = sm.nextu();
            }
            use std::arch::x86_64::_mm512_loadu_si512;
            (
                _mm512_loadu_si512(sv[0].as_ptr() as *const _),
                _mm512_loadu_si512(sv[1].as_ptr() as *const _),
                _mm512_loadu_si512(sv[2].as_ptr() as *const _),
                _mm512_loadu_si512(sv[3].as_ptr() as *const _),
            )
        }};
    }

    let (mut s0_0, mut s1_0, mut s2_0, mut s3_0) = make_state_vec!(0);
    let (mut s0_1, mut s1_1, mut s2_1, mut s3_1) = make_state_vec!(1);
    let (mut s0_2, mut s1_2, mut s2_2, mut s3_2) = make_state_vec!(2);
    let (mut s0_3, mut s1_3, mut s2_3, mut s3_3) = make_state_vec!(3);

    // xoshiro256++ step: res = rotl(s0+s3,23)+s0; update state in place.
    // All 4 groups' sums and rotates are issued together to maximize port-0 utilization.
    macro_rules! step4 {
        () => {{
            use std::arch::x86_64::{
                _mm512_add_epi64, _mm512_rol_epi64, _mm512_slli_epi64, _mm512_xor_si512,
            };
            // Issue 4 sums + 4 rotates together (4 port-0 ROL ops at once)
            let sum0 = _mm512_add_epi64(s0_0, s3_0);
            let sum1 = _mm512_add_epi64(s0_1, s3_1);
            let sum2 = _mm512_add_epi64(s0_2, s3_2);
            let sum3 = _mm512_add_epi64(s0_3, s3_3);
            let rot0 = _mm512_rol_epi64(sum0, 23);
            let rot1 = _mm512_rol_epi64(sum1, 23);
            let rot2 = _mm512_rol_epi64(sum2, 23);
            let rot3 = _mm512_rol_epi64(sum3, 23);
            let res0 = _mm512_add_epi64(rot0, s0_0);
            let res1 = _mm512_add_epi64(rot1, s0_1);
            let res2 = _mm512_add_epi64(rot2, s0_2);
            let res3 = _mm512_add_epi64(rot3, s0_3);
            // Compute t for all groups (slli = port 0,5)
            let t0 = _mm512_slli_epi64(s1_0, 17);
            let t1 = _mm512_slli_epi64(s1_1, 17);
            let t2 = _mm512_slli_epi64(s1_2, 17);
            let t3 = _mm512_slli_epi64(s1_3, 17);
            // State update (all XORs, ports 0,1,5)
            s2_0 = _mm512_xor_si512(s2_0, s0_0);
            s2_1 = _mm512_xor_si512(s2_1, s0_1);
            s2_2 = _mm512_xor_si512(s2_2, s0_2);
            s2_3 = _mm512_xor_si512(s2_3, s0_3);
            s3_0 = _mm512_xor_si512(s3_0, s1_0);
            s3_1 = _mm512_xor_si512(s3_1, s1_1);
            s3_2 = _mm512_xor_si512(s3_2, s1_2);
            s3_3 = _mm512_xor_si512(s3_3, s1_3);
            s1_0 = _mm512_xor_si512(s1_0, s2_0);
            s1_1 = _mm512_xor_si512(s1_1, s2_1);
            s1_2 = _mm512_xor_si512(s1_2, s2_2);
            s1_3 = _mm512_xor_si512(s1_3, s2_3);
            s0_0 = _mm512_xor_si512(s0_0, s3_0);
            s0_1 = _mm512_xor_si512(s0_1, s3_1);
            s0_2 = _mm512_xor_si512(s0_2, s3_2);
            s0_3 = _mm512_xor_si512(s0_3, s3_3);
            s2_0 = _mm512_xor_si512(s2_0, t0);
            s2_1 = _mm512_xor_si512(s2_1, t1);
            s2_2 = _mm512_xor_si512(s2_2, t2);
            s2_3 = _mm512_xor_si512(s2_3, t3);
            // Final 4 ROLs for s3 (4 more port-0 ops)
            s3_0 = _mm512_rol_epi64(s3_0, 45);
            s3_1 = _mm512_rol_epi64(s3_1, 45);
            s3_2 = _mm512_rol_epi64(s3_2, 45);
            s3_3 = _mm512_rol_epi64(s3_3, 45);
            (res0, res1, res2, res3)
        }};
    }

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(32);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            use std::arch::x86_64::_mm512_stream_si512;

            let (r0, r1, r2, r3) = step4!();
            let p = dst.as_mut_ptr();
            _mm512_stream_si512(p as *mut _, r0);
            _mm512_stream_si512(p.add(8) as *mut _, r1);
            _mm512_stream_si512(p.add(16) as *mut _, r2);
            _mm512_stream_si512(p.add(24) as *mut _, r3);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3) = step4!();
            let p = dst.as_mut_ptr();
            _mm512_storeu_si512(p as *mut _, r0);
            _mm512_storeu_si512(p.add(8) as *mut _, r1);
            _mm512_storeu_si512(p.add(16) as *mut _, r2);
            _mm512_storeu_si512(p.add(24) as *mut _, r3);
        }
    }

    // Handle remainder (< 32 elements)
    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0u64; 32];
        let (r0, r1, r2, r3) = step4!();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut _, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut _, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut _, r3);
        for (j, v) in rem.iter_mut().enumerate() {
            *v = tmp[j];
        }
    }
}

/// Fills `out[0..count]` with raw `u64` random values.
/// Uses AVX-512 8-stream SoA with 4-way interleaving and rayon parallelism.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_next_u64s(ptr: *mut Xoshiro256Ssx2, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut s_arr = [0u64; 8];
        _mm512_storeu_si512(s_arr.as_mut_ptr() as *mut _, rng.s);
        let base_seed = s_arr[0]
            .wrapping_add(s_arr[1])
            .wrapping_add(s_arr[2])
            .wrapping_add(s_arr[3]);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOSHIRO256SSX2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                xoshiro256ssx2_next_u64s_chunk(chunk_idx, chunk, base_seed);
            });

        // Advance RNG state so next call produces a different sequence
        let new_seed = SplitMix64::compute(
            base_seed.wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        );
        *rng = Xoshiro256Ssx2::new(new_seed);
    }
}

/// Creates a new heap-allocated `Xoshiro256Ss` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256ss_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_new(seed: u64) -> *mut Xoshiro256Ss {
    Box::into_raw(Box::new(Xoshiro256Ss::new(seed)))
}
/// Frees a `Xoshiro256Ss` instance previously created by [`xoshiro256ss_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_free(ptr: *mut Xoshiro256Ss) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256SS_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_u64s(ptr: *mut Xoshiro256Ss, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_f64s(ptr: *mut Xoshiro256Ss, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_rand_i64s(
    ptr: *mut Xoshiro256Ss,
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
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_rand_f64s(
    ptr: *mut Xoshiro256Ss,
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
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
    }
}

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

/// Creates a new heap-allocated `Threefish256` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`threefish256_free`].
#[unsafe(no_mangle)]
pub extern "C" fn threefish256_new(seed: u64) -> *mut Threefish256 {
    Box::into_raw(Box::new(Threefish256::new(seed)))
}
/// Frees a `Threefish256` instance previously created by [`threefish256_new`].
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
