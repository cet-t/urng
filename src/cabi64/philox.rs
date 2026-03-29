use crate::rng64::Philox64;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::slice::from_raw_parts_mut;

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
