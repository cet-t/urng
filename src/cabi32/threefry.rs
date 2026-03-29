use crate::rng32::{Threefry32x2, Threefry32x4};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::slice::from_raw_parts_mut;

/// Creates a new `Threefry32x4` instance.
/// The caller is responsible for freeing the memory using `threefry32x4_free`.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_new(seed: u32) -> *mut Threefry32x4 {
    Box::into_raw(Box::new(Threefry32x4::new(seed)))
}

/// Frees the memory of a `Threefry32x4` instance.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_free(ptr: *mut Threefry32x4) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const THREEFRY32_PAR_CHUNK: usize = 4096;

/// Fills the output buffer with the next random `u32` values.
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_u32s(ptr: *mut Threefry32x4, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    dst[2] = result[2];
                    dst[3] = result[3];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_f32s(ptr: *mut Threefry32x4, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = result[0] as f32 * SCALE;
                    dst[1] = result[1] as f32 * SCALE;
                    dst[2] = result[2] as f32 * SCALE;
                    dst[3] = result[3] as f32 * SCALE;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * SCALE;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
/// Fills the output buffer with random `i32` values in the range [min, max].
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_i32s(
    ptr: *mut Threefry32x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((result[2] as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((result[3] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}
/// Fills the output buffer with random `f32` values in the range [min, max).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_f32s(
    ptr: *mut Threefry32x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let tw = rng.tw;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(THREEFRY32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    dst[2] = (result[2] as f32 * scale_val) * range_val + min;
                    dst[3] = (result[3] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [current_c64 as u32, (current_c64 >> 32) as u32, c0[2], c0[3]];

                    let result = Threefry32x4::compute(c, &k, &tw);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
        if new_c64 < c0_64 {
            let (n_c2, ovf3) = rng.c[2].overflowing_add(1);
            rng.c[2] = n_c2;
            if ovf3 {
                rng.c[3] = rng.c[3].wrapping_add(1);
            }
        }
    }
}

/// Creates a new `Threefry32x2` instance.
/// The caller is responsible for freeing the memory using `threefry32x2_free`.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_new(seed: u32) -> *mut Threefry32x2 {
    Box::into_raw(Box::new(Threefry32x2::new(seed)))
}

/// Frees the memory of a `Threefry32x2` instance.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_free(ptr: *mut Threefry32x2) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const THREEFRY32X2_PAR_CHUNK: usize = 4096;

/// Fills the output buffer with the next random `u32` values.
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_u32s(ptr: *mut Threefry32x2, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = result[j];
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_f32s(ptr: *mut Threefry32x2, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = result[0] as f32 * SCALE;
                    dst[1] = result[1] as f32 * SCALE;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * SCALE;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_rand_i32s(
    ptr: *mut Threefry32x2,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_rand_f32s(
    ptr: *mut Threefry32x2,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_offset = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
                let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
                let c64_start = c0_64.wrapping_add(chunk_base_offset);

                let mut chunks_exact = chunk.chunks_exact_mut(2);
                let mut b_offset = 0u64;

                for dst in chunks_exact.by_ref() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let current_c64 = c64_start.wrapping_add(b_offset);
                    let c = [(current_c64 as u32), ((current_c64 >> 32) as u32)];

                    let result = Threefry32x2::compute(c, &k);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 1) / 2;
        let c0_64 = (rng.c[0] as u64) | ((rng.c[1] as u64) << 32);
        let new_c64 = c0_64.wrapping_add(num_blocks as u64);
        rng.c[0] = new_c64 as u32;
        rng.c[1] = (new_c64 >> 32) as u32;
    }
}
