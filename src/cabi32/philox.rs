use crate::dispatch_simd;
use crate::rng32::{
    PHILOX32x4x4_CHUNK_RATIO, PHILOX32x4x4_PAR_CHUNK, PHILOX32x4x4_SHIFT, PHILOX32x16,
    PHILOX32x16_SHIFT, Philox32, Philox32x4, Philox32x4x4,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::arch::x86_64::*;
use std::slice::from_raw_parts_mut;

/// Creates a new `Philox32x4` instance.
/// The caller is responsible for freeing the memory using `philox32x4_free`.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_new(seed: u32) -> *mut Philox32x4 {
    Box::into_raw(Box::new(Philox32x4::new(seed)))
}

/// Frees the memory of a `Philox32x4` instance.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_free(ptr: *mut Philox32x4) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const PHILOX32_PAR_CHUNK: usize = 4096;

/// Fills the output buffer with the next random `u32` values.
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_next_u32s(ptr: *mut Philox32x4, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = result[0];
                    dst[1] = result[1];
                    dst[2] = result[2];
                    dst[3] = result[3];
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for i in 0..rem.len() {
                        rem[i] = result[i];
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_next_f32s(ptr: *mut Philox32x4, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale = 1.0f32 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = result[0] as f32 * scale;
                    dst[1] = result[1] as f32 * scale;
                    dst[2] = result[2] as f32 * scale;
                    dst[3] = result[3] as f32 * scale;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = result[j] as f32 * scale;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_rand_i32s(
    ptr: *mut Philox32x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = ((result[0] as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((result[1] as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((result[2] as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((result[3] as u64 * range) >> 32) as i32 + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4_rand_f32s(
    ptr: *mut Philox32x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;

        buffer
            .par_chunks_mut(PHILOX32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_base_block = (chunk_idx * PHILOX32_PAR_CHUNK) / 4;
                let mut chunks_exact = chunk.chunks_exact_mut(4);
                let mut b_offset = 0u32;

                for dst in chunks_exact.by_ref() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    dst[0] = (result[0] as f32 * scale_val) * range_val + min;
                    dst[1] = (result[1] as f32 * scale_val) * range_val + min;
                    dst[2] = (result[2] as f32 * scale_val) * range_val + min;
                    dst[3] = (result[3] as f32 * scale_val) * range_val + min;
                    b_offset += 1;
                }

                let rem = chunks_exact.into_remainder();
                if !rem.is_empty() {
                    let mut c = c0;
                    let (new_c0, overflow) =
                        c[0].0.overflowing_add((chunk_base_block as u32) + b_offset);
                    c[0].0 = new_c0;
                    if overflow {
                        c[1].0 = c[1].0.wrapping_add(1);
                    }

                    let result = Philox32x4::compute(c, k);
                    for j in 0..rem.len() {
                        rem[j] = (result[j] as f32 * scale_val) * range_val + min;
                    }
                }
            });

        let num_blocks = (count + 3) / 4;
        let (n_c0, overflow) = rng.c[0].0.overflowing_add(num_blocks as u32);
        rng.c[0].0 = n_c0;
        if overflow {
            let (n_c1, ovf2) = rng.c[1].0.overflowing_add(1);
            rng.c[1].0 = n_c1;
            if ovf2 {
                rng.c[2].0 = rng.c[2].0.wrapping_add(1);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
fn philox32x4x4_compute_vec(mut x: __m512i, mut key: __m512i, m: __m512i, w: __m512i) -> __m512i {
    macro_rules! round {
        () => {{
            let prod = _mm512_mul_epu32(x, m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64(x, 32);
            x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, key));
            key = _mm512_add_epi32(key, w);
        }};
    }

    round!();
    round!();
    round!();
    round!();
    round!();
    round!();
    round!();
    round!();
    round!();
    round!();

    let _ = key;
    x
}

/// Creates a new `Philox32x4x4` instance.
/// The caller is responsible for freeing the memory using `philox32x4x4_free`.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_new(seed: u32) -> *mut Philox32x4x4 {
    unsafe { Box::into_raw(Box::new(Philox32x4x4::new(seed))) }
}

/// Frees the memory of a `Philox32x4x4` instance.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_free(ptr: *mut Philox32x4x4) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
#[allow(unused_assignments)]
unsafe fn philox32x4x4_next_u32s_chunk(
    chunk_idx: usize,
    chunk: &mut [u32],
    c: __m512i,
    k: __m512i,
    one: __m512i,
) {
    // 8-way interleaved Philox rounds: issue 8 multiplies to fully hide
    // the 5-cycle mul latency, then complete shuffle/xor/key-advance per block.
    macro_rules! round8 {
        ($x0:ident, $k0:ident, $x1:ident, $k1:ident,
         $x2:ident, $k2:ident, $x3:ident, $k3:ident,
         $x4:ident, $k4:ident, $x5:ident, $k5:ident,
         $x6:ident, $k6:ident, $x7:ident, $k7:ident, $m:ident, $w:ident) => {{
            let p0 = _mm512_mul_epu32($x0, $m);
            let p1 = _mm512_mul_epu32($x1, $m);
            let p2 = _mm512_mul_epu32($x2, $m);
            let p3 = _mm512_mul_epu32($x3, $m);
            let p4 = _mm512_mul_epu32($x4, $m);
            let p5 = _mm512_mul_epu32($x5, $m);
            let p6 = _mm512_mul_epu32($x6, $m);
            let p7 = _mm512_mul_epu32($x7, $m);

            let s0 = _mm512_shuffle_epi32(p0, 0x1B);
            let xs0 = _mm512_srli_epi64($x0, 32);
            $x0 = _mm512_xor_epi32(s0, _mm512_xor_epi32(xs0, $k0));
            $k0 = _mm512_add_epi32($k0, $w);

            let s1 = _mm512_shuffle_epi32(p1, 0x1B);
            let xs1 = _mm512_srli_epi64($x1, 32);
            $x1 = _mm512_xor_epi32(s1, _mm512_xor_epi32(xs1, $k1));
            $k1 = _mm512_add_epi32($k1, $w);

            let s2 = _mm512_shuffle_epi32(p2, 0x1B);
            let xs2 = _mm512_srli_epi64($x2, 32);
            $x2 = _mm512_xor_epi32(s2, _mm512_xor_epi32(xs2, $k2));
            $k2 = _mm512_add_epi32($k2, $w);

            let s3 = _mm512_shuffle_epi32(p3, 0x1B);
            let xs3 = _mm512_srli_epi64($x3, 32);
            $x3 = _mm512_xor_epi32(s3, _mm512_xor_epi32(xs3, $k3));
            $k3 = _mm512_add_epi32($k3, $w);

            let s4 = _mm512_shuffle_epi32(p4, 0x1B);
            let xs4 = _mm512_srli_epi64($x4, 32);
            $x4 = _mm512_xor_epi32(s4, _mm512_xor_epi32(xs4, $k4));
            $k4 = _mm512_add_epi32($k4, $w);

            let s5 = _mm512_shuffle_epi32(p5, 0x1B);
            let xs5 = _mm512_srli_epi64($x5, 32);
            $x5 = _mm512_xor_epi32(s5, _mm512_xor_epi32(xs5, $k5));
            $k5 = _mm512_add_epi32($k5, $w);

            let s6 = _mm512_shuffle_epi32(p6, 0x1B);
            let xs6 = _mm512_srli_epi64($x6, 32);
            $x6 = _mm512_xor_epi32(s6, _mm512_xor_epi32(xs6, $k6));
            $k6 = _mm512_add_epi32($k6, $w);

            let s7 = _mm512_shuffle_epi32(p7, 0x1B);
            let xs7 = _mm512_srli_epi64($x7, 32);
            $x7 = _mm512_xor_epi32(s7, _mm512_xor_epi32(xs7, $k7));
            $k7 = _mm512_add_epi32($k7, $w);
        }};
    }

    macro_rules! round4 {
        ($x0:ident, $k0:ident, $x1:ident, $k1:ident,
         $x2:ident, $k2:ident, $x3:ident, $k3:ident, $m:ident, $w:ident) => {{
            let p0 = _mm512_mul_epu32($x0, $m);
            let p1 = _mm512_mul_epu32($x1, $m);
            let p2 = _mm512_mul_epu32($x2, $m);
            let p3 = _mm512_mul_epu32($x3, $m);

            let s0 = _mm512_shuffle_epi32(p0, 0x1B);
            let xs0 = _mm512_srli_epi64($x0, 32);
            $x0 = _mm512_xor_epi32(s0, _mm512_xor_epi32(xs0, $k0));
            $k0 = _mm512_add_epi32($k0, $w);

            let s1 = _mm512_shuffle_epi32(p1, 0x1B);
            let xs1 = _mm512_srli_epi64($x1, 32);
            $x1 = _mm512_xor_epi32(s1, _mm512_xor_epi32(xs1, $k1));
            $k1 = _mm512_add_epi32($k1, $w);

            let s2 = _mm512_shuffle_epi32(p2, 0x1B);
            let xs2 = _mm512_srli_epi64($x2, 32);
            $x2 = _mm512_xor_epi32(s2, _mm512_xor_epi32(xs2, $k2));
            $k2 = _mm512_add_epi32($k2, $w);

            let s3 = _mm512_shuffle_epi32(p3, 0x1B);
            let xs3 = _mm512_srli_epi64($x3, 32);
            $x3 = _mm512_xor_epi32(s3, _mm512_xor_epi32(xs3, $k3));
            $k3 = _mm512_add_epi32($k3, $w);
        }};
    }

    macro_rules! philox10_single {
        ($x:ident, $key:ident, $m:ident, $w:ident) => {{
            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
            $key = _mm512_add_epi32($key, $w);

            let prod = _mm512_mul_epu32($x, $m);
            let shuf = _mm512_shuffle_epi32(prod, 0x1B);
            let x_shift = _mm512_srli_epi64($x, 32);
            $x = _mm512_xor_epi32(shuf, _mm512_xor_epi32(x_shift, $key));
        }};
    }

    let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
    let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);
    let offset = (chunk_idx as u128) << PHILOX32x4x4_SHIFT;

    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    // Counter step vectors (carry-free: 64-bit lower counter won't overflow)
    let two = _mm512_set1_epi64(2);
    let three = _mm512_set1_epi64(3);
    let four = _mm512_set1_epi64(4);
    let five = _mm512_set1_epi64(5);
    let six = _mm512_set1_epi64(6);
    let seven = _mm512_set1_epi64(7);
    let eight = _mm512_set1_epi64(8);

    // --- 8-way interleaved main loop (128 u32s = 512 bytes per iteration) ---
    let ptr = chunk.as_mut_ptr();
    let len = chunk.len();
    let full8 = len / (PHILOX32x16 * 8);
    let mut p = ptr;

    for _ in 0..full8 {
        // TLB prefetch: warm the page 2KB ahead (half page) so page walks
        // complete before streaming stores need the TLB entry.
        unsafe { _mm_prefetch(p.add(PHILOX32x16 * 8 * 2) as *const i8, _MM_HINT_T2) };

        let c0 = c;
        let c1 = _mm512_mask_add_epi64(c, 0x55, c, one);
        let c2 = _mm512_mask_add_epi64(c, 0x55, c, two);
        let c3 = _mm512_mask_add_epi64(c, 0x55, c, three);
        let c4 = _mm512_mask_add_epi64(c, 0x55, c, four);
        let c5 = _mm512_mask_add_epi64(c, 0x55, c, five);
        let c6 = _mm512_mask_add_epi64(c, 0x55, c, six);
        let c7 = _mm512_mask_add_epi64(c, 0x55, c, seven);
        c = _mm512_mask_add_epi64(c, 0x55, c, eight);

        let mut x0 = c0;
        let mut x1 = c1;
        let mut x2 = c2;
        let mut x3 = c3;
        let mut x4 = c4;
        let mut x5 = c5;
        let mut x6 = c6;
        let mut x7 = c7;
        let mut k0 = k;
        let mut k1 = k;
        let mut k2 = k;
        let mut k3 = k;
        let mut k4 = k;
        let mut k5 = k;
        let mut k6 = k;
        let mut k7 = k;

        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );
        round8!(
            x0, k0, x1, k1, x2, k2, x3, k3, x4, k4, x5, k5, x6, k6, x7, k7, m, w
        );

        unsafe {
            _mm512_stream_si512(p as *mut _, x0);
            _mm512_stream_si512(p.add(PHILOX32x16) as *mut _, x1);
            _mm512_stream_si512(p.add(PHILOX32x16 * 2) as *mut _, x2);
            _mm512_stream_si512(p.add(PHILOX32x16 * 3) as *mut _, x3);
            _mm512_stream_si512(p.add(PHILOX32x16 * 4) as *mut _, x4);
            _mm512_stream_si512(p.add(PHILOX32x16 * 5) as *mut _, x5);
            _mm512_stream_si512(p.add(PHILOX32x16 * 6) as *mut _, x6);
            _mm512_stream_si512(p.add(PHILOX32x16 * 7) as *mut _, x7);
        }
        p = unsafe { p.add(PHILOX32x16 * 8) };
    }

    // --- 4-way remainder (0-7 blocks of 16) ---
    let remaining = len - full8 * PHILOX32x16 * 8;
    let full4 = remaining / (PHILOX32x16 * 4);
    for _ in 0..full4 {
        let c0 = c;
        let c1 = _mm512_mask_add_epi64(c, 0x55, c, one);
        let c2 = _mm512_mask_add_epi64(c, 0x55, c, two);
        let c3 = _mm512_mask_add_epi64(c, 0x55, c, three);
        c = _mm512_mask_add_epi64(c, 0x55, c, four);

        let mut x0 = c0;
        let mut x1 = c1;
        let mut x2 = c2;
        let mut x3 = c3;
        let mut k0 = k;
        let mut k1 = k;
        let mut k2 = k;
        let mut k3 = k;

        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);
        round4!(x0, k0, x1, k1, x2, k2, x3, k3, m, w);

        unsafe {
            _mm512_stream_si512(p as *mut _, x0);
            _mm512_stream_si512(p.add(PHILOX32x16) as *mut _, x1);
            _mm512_stream_si512(p.add(PHILOX32x16 * 2) as *mut _, x2);
            _mm512_stream_si512(p.add(PHILOX32x16 * 3) as *mut _, x3);
        }
        p = unsafe { p.add(PHILOX32x16 * 4) };
    }

    // --- Single-block remainder ---
    let remaining2 = remaining - full4 * PHILOX32x16 * 4;
    let full1 = remaining2 / PHILOX32x16;
    for _ in 0..full1 {
        let mut x = c;
        let mut key_local = k;
        philox10_single!(x, key_local, m, w);
        unsafe { _mm512_stream_si512(p as *mut _, x) };
        c = _mm512_mask_add_epi64(c, 0x55, c, one);
        p = unsafe { p.add(PHILOX32x16) };
    }

    // --- Partial tail ---
    let tail = remaining2 - full1 * PHILOX32x16;
    if tail > 0 {
        let mut x = c;
        let mut key_local = k;
        philox10_single!(x, key_local, m, w);
        let mut tmp = [0u32; PHILOX32x16];
        unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, x) };
        for j in 0..tail {
            unsafe { *p.add(j) = tmp[j] };
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_next_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c0: __m512i,
    k: __m512i,
    one: __m512i,
    scale: __m512,
) {
    let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
    let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);
    let zero = _mm512_setzero_si512();
    let offset = chunk_idx as u128 * PHILOX32x4x4_CHUNK_RATIO;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c0) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_mul_ps(v_f32, scale);
            unsafe { _mm512_stream_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_mul_ps(v_f32, scale);
            unsafe { _mm512_storeu_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        let v_res = _mm512_mul_ps(v_f32, scale);
        let mut tmp_f32 = [0f32; 16];
        unsafe { _mm512_storeu_ps(tmp_f32.as_mut_ptr() as *mut _, v_res) };
        for j in 0..rem.len() {
            rem[j] = tmp_f32[j];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_rand_i32s_chunk(
    chunk_idx: usize,
    chunk: &mut [i32],
    c: __m512i,
    k: __m512i,
    one: __m512i,
    v_range: __m512i,
    v_min: __m512i,
    merge_mask: u16,
) {
    let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
    let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);
    let zero = _mm512_setzero_si512();
    let offset = (chunk_idx as u128) << PHILOX32x4x4_SHIFT;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0; // N % 64 == 0
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);

            let prod_even = _mm512_mul_epu32(v_u32, v_range);
            let res_even = _mm512_srli_epi64(prod_even, 32);

            let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
            let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

            let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
            let v_res = _mm512_add_epi32(merged, v_min);
            unsafe { _mm512_stream_si512(dst.as_mut_ptr() as *mut _, v_res) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);

            let prod_even = _mm512_mul_epu32(v_u32, v_range);
            let res_even = _mm512_srli_epi64(prod_even, 32);

            let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
            let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

            let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
            let v_res = _mm512_add_epi32(merged, v_min);

            unsafe { _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, v_res) };

            // +1
            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
        let prod_even = _mm512_mul_epu32(v_u32, v_range);
        let res_even = _mm512_srli_epi64(prod_even, 32);

        let v_u32_shifted = _mm512_srli_epi64(v_u32, 32);
        let prod_odd = _mm512_mul_epu32(v_u32_shifted, v_range);

        let merged = _mm512_mask_blend_epi32(merge_mask, res_even, prod_odd);
        let v_res = _mm512_add_epi32(merged, v_min);

        let mut tmp_res = [0i32; 16];
        unsafe { _mm512_storeu_si512(tmp_res.as_mut_ptr() as *mut _, v_res) };

        for j in 0..rem.len() {
            rem[j] = tmp_res[j];
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn philox32x4x4_rand_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c: __m512i,
    k: __m512i,
    one: __m512i,
    v_mult: __m512,
    v_min: __m512,
) {
    let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
    let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);
    let zero = _mm512_setzero_si512();
    let offset = chunk_idx as u128 * PHILOX32x4x4_CHUNK_RATIO;
    let mut c_array = [0u128; 4];
    unsafe { _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, c) };
    for i in 0..4 {
        c_array[i] = c_array[i].wrapping_add(offset);
    }
    let mut c = unsafe { _mm512_loadu_si512(c_array.as_ptr() as *const _) };

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(PHILOX32x16);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
            unsafe { _mm512_stream_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
            let v_f32 = _mm512_cvtepu32_ps(v_u32);
            let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
            unsafe { _mm512_storeu_ps(dst.as_mut_ptr() as *mut f32, v_res) };

            let next_c = _mm512_mask_add_epi64(c, 0x55, c, one);
            let eq_zero_mask = _mm512_cmpeq_epi64_mask(next_c, zero);
            let carry_mask = (eq_zero_mask & 0x55) << 1;
            c = _mm512_mask_add_epi64(next_c, carry_mask, next_c, one);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let v_u32 = philox32x4x4_compute_vec(c, k, m, w);
        let v_f32 = _mm512_cvtepu32_ps(v_u32);
        let v_res = _mm512_fmadd_ps(v_f32, v_mult, v_min);
        let mut tmp_f32 = [0f32; 16];
        unsafe { _mm512_storeu_ps(tmp_f32.as_mut_ptr() as *mut _, v_res) };
        for j in 0..rem.len() {
            rem[j] = tmp_f32[j];
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
/// This function utilizes AVX-512 SIMD and parallel processing for high throughput.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_next_u32s(ptr: *mut Philox32x4x4, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);

        // Align output to 64 bytes for streaming stores
        let misalign_bytes = (out as usize) & 63;
        let head_elems = if misalign_bytes == 0 {
            0
        } else {
            ((64 - misalign_bytes) / 4).min(count)
        };

        // Process unaligned head (at most 15 elements via one Philox block)
        if head_elems > 0 {
            let m = _mm512_set1_epi64(0xCD9E8D57_D2511F53u64 as i64);
            let w = _mm512_set1_epi64(0xBB67AE85_9E3779B9u64 as i64);
            let x = philox32x4x4_compute_vec(c, k, m, w);
            let mut tmp = [0u32; PHILOX32x16];
            _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, x);
            for i in 0..head_elems {
                *out.add(i) = tmp[i];
            }
        }

        let body_count = count - head_elems;
        if body_count > 0 {
            let body_ptr = out.add(head_elems);
            let body_buffer = from_raw_parts_mut(body_ptr, body_count);

            // Advance counter past head block
            let c_body = if head_elems > 0 {
                let mut c_arr = [0u128; 4];
                _mm512_storeu_si512(c_arr.as_mut_ptr() as *mut _, c);
                for i in 0..4 {
                    c_arr[i] = c_arr[i].wrapping_add(1);
                }
                _mm512_loadu_si512(c_arr.as_ptr() as *const _)
            } else {
                c
            };

            body_buffer
                .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    philox32x4x4_next_u32s_chunk(chunk_idx, chunk, c_body, k, one)
                });
        }

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_next_f32s(ptr: *mut Philox32x4x4, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c0 = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);
        let scale = _mm512_set1_ps(1.0f32 / (u32::MAX as f32 + 1.0));

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_next_f32s_chunk(chunk_idx, chunk, c0, k, one, scale);
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_rand_i32s(
    ptr: *mut Philox32x4x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);
        let range = (max as i64 - min as i64 + 1) as u64;

        let v_range = _mm512_set1_epi64(range as i64);
        let v_min = _mm512_set1_epi32(min);
        let merge_mask = 0b1010101010101010;

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_rand_i32s_chunk(
                    chunk_idx, chunk, c, k, one, v_range, v_min, merge_mask,
                );
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}
/// Fills the output buffer with random `f32` values in the range [min, max).
/// This function utilizes AVX-512 SIMD and parallel processing for high throughput.
#[unsafe(no_mangle)]
pub extern "C" fn philox32x4x4_rand_f32s(
    ptr: *mut Philox32x4x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let c = rng.c;
        let k = rng.k;
        let one = _mm512_set1_epi64(1);

        let scale_val = 1.0f32 / (u32::MAX as f32 + 1.0);
        let range_val = max - min;
        let scale_mul_range = scale_val * range_val;

        let v_mult = _mm512_set1_ps(scale_mul_range);
        let v_min = _mm512_set1_ps(min);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PHILOX32x4x4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                philox32x4x4_rand_f32s_chunk(chunk_idx, chunk, c, k, one, v_mult, v_min);
            });

        let num_blocks = ((count + PHILOX32x16 - 1) >> PHILOX32x16_SHIFT) as u128;
        let mut c_array = [0u128; 4];
        _mm512_storeu_si512(c_array.as_mut_ptr() as *mut _, rng.c);
        for i in 0..4 {
            c_array[i] = c_array[i].wrapping_add(num_blocks);
        }
        rng.c = _mm512_loadu_si512(c_array.as_ptr() as *const _);
    }
}

/// Creates a new `Philox32` instance, dispatching to AVX-512 or scalar implementation.
/// The caller is responsible for freeing the memory using `philox32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn philox32_new(seed: u32) -> *mut Philox32 {
    dispatch_simd!(Philox32, philox32x4_new, philox32x4x4_new, seed)
}
/// Frees the memory of a `Philox32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn philox32_free(ptr: *mut Philox32) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_free,
        philox32x4x4_free,
        ptr
    )
}
/// Fills the output buffer with the next random `u32` values using the best available implementation.
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_u32s(ptr: *mut Philox32, out: *mut u32, count: usize) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_next_u32s,
        philox32x4x4_next_u32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn philox32_next_f32s(ptr: *mut Philox32, out: *mut f32, count: usize) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_next_f32s,
        philox32x4x4_next_f32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_i32s(
    ptr: *mut Philox32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_rand_i32s,
        philox32x4x4_rand_i32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn philox32_rand_f32s(
    ptr: *mut Philox32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    dispatch_simd!(
        Philox32x4x4,
        Philox32x4,
        philox32x4_rand_f32s,
        philox32x4x4_rand_f32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
