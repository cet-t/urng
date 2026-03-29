use crate::dispatch_simd;
use crate::rng32::{SQUARES32x8, Squares32, Squares32Simd, Squares32x8};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::arch::x86_64::*;
use std::slice::from_raw_parts_mut;

#[allow(non_upper_case_globals)]
const SQUARES32x8_PAR_CHUNK: usize = 8192;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn squares32x8_next_u32s_chunk(
    chunk_idx: usize,
    chunk: &mut [u32],
    c0: u64,
    k: __m512i,
    lane_offsets: __m512i,
) {
    unsafe {
        let c_start = c0.wrapping_add((chunk_idx * SQUARES32x8_PAR_CHUNK) as u64);
        let c_vec = _mm512_add_epi64(_mm512_set1_epi64(c_start as i64), lane_offsets);
        let kx1 = k;
        let kx8 = _mm512_slli_epi64(k, 3);
        let kx32 = _mm512_slli_epi64(k, 5);
        let mut y0 = _mm512_mullo_epi64(c_vec, kx1);

        let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
        let mut chunks32 = chunk.chunks_exact_mut(SQUARES32x8 * 4);

        if is_aligned {
            for dst in chunks32.by_ref() {
                let y1 = _mm512_add_epi64(y0, kx8);
                let y2 = _mm512_add_epi64(y1, kx8);
                let y3 = _mm512_add_epi64(y2, kx8);

                let v0 = Squares32x8::compute_yz(y0, _mm512_add_epi64(y0, kx1));
                let v1 = Squares32x8::compute_yz(y1, _mm512_add_epi64(y1, kx1));
                let v2 = Squares32x8::compute_yz(y2, _mm512_add_epi64(y2, kx1));
                let v3 = Squares32x8::compute_yz(y3, _mm512_add_epi64(y3, kx1));

                let res01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
                let res23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3);

                _mm512_stream_si512(dst.as_mut_ptr() as *mut _, res01);
                _mm512_stream_si512(dst[16..].as_mut_ptr() as *mut _, res23);

                y0 = _mm512_add_epi64(y0, kx32);
            }
        } else {
            for dst in chunks32.by_ref() {
                let y1 = _mm512_add_epi64(y0, kx8);
                let y2 = _mm512_add_epi64(y1, kx8);
                let y3 = _mm512_add_epi64(y2, kx8);

                let v0 = Squares32x8::compute_yz(y0, _mm512_add_epi64(y0, kx1));
                let v1 = Squares32x8::compute_yz(y1, _mm512_add_epi64(y1, kx1));
                let v2 = Squares32x8::compute_yz(y2, _mm512_add_epi64(y2, kx1));
                let v3 = Squares32x8::compute_yz(y3, _mm512_add_epi64(y3, kx1));

                let res01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
                let res23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3);

                _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, res01);
                _mm512_storeu_si512(dst[16..].as_mut_ptr() as *mut _, res23);

                y0 = _mm512_add_epi64(y0, kx32);
            }
        }

        let rem = chunks32.into_remainder();
        let mut rem_chunks8 = rem.chunks_exact_mut(SQUARES32x8);
        for dst in rem_chunks8.by_ref() {
            let v = Squares32x8::compute_yz(y0, _mm512_add_epi64(y0, kx1));
            _mm256_storeu_si256(dst.as_mut_ptr() as *mut _, v);
            y0 = _mm512_add_epi64(y0, kx8);
        }
        let final_rem = rem_chunks8.into_remainder();
        if !final_rem.is_empty() {
            let v = Squares32x8::compute_yz(y0, _mm512_add_epi64(y0, kx1));
            let mut tmp = [0u32; SQUARES32x8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, v);
            for j in 0..final_rem.len() {
                final_rem[j] = tmp[j];
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn squares32x8_next_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c0: u64,
    k: __m512i,
    lane_offsets: __m512i,
    k_step: __m512i,
    k_step2: __m512i,
    k_step3: __m512i,
    k_step4: __m512i,
) {
    unsafe {
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
        let vscale = _mm512_set1_ps(SCALE);
        let c_start = c0.wrapping_add((chunk_idx * SQUARES32x8_PAR_CHUNK) as u64);
        let mut c_vec = _mm512_add_epi64(_mm512_set1_epi64(c_start as i64), lane_offsets);

        let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
        let mut chunks32 = chunk.chunks_exact_mut(SQUARES32x8 * 4);

        for dst in chunks32.by_ref() {
            let v0 = Squares32x8::compute(c_vec, k);
            let v1 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step), k);
            let v2 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step2), k);
            let v3 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step3), k);

            let res01 = _mm512_cvtepu32_ps(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1));
            let res23 = _mm512_cvtepu32_ps(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3));

            let f01 = _mm512_mul_ps(res01, vscale);
            let f23 = _mm512_mul_ps(res23, vscale);

            if is_aligned {
                _mm512_stream_ps(dst.as_mut_ptr(), f01);
                _mm512_stream_ps(dst[16..].as_mut_ptr(), f23);
            } else {
                _mm512_storeu_ps(dst.as_mut_ptr(), f01);
                _mm512_storeu_ps(dst[16..].as_mut_ptr(), f23);
            }

            c_vec = _mm512_add_epi64(c_vec, k_step4);
        }

        let rem = chunks32.into_remainder();
        let mut rem_chunks8 = rem.chunks_exact_mut(SQUARES32x8);
        for dst in rem_chunks8.by_ref() {
            let v = Squares32x8::compute(c_vec, k);
            let mut result = [0u32; SQUARES32x8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut _, v);
            for j in 0..SQUARES32x8 {
                dst[j] = result[j] as f32 * SCALE;
            }
            c_vec = _mm512_add_epi64(c_vec, k_step);
        }
        let final_rem = rem_chunks8.into_remainder();
        if !final_rem.is_empty() {
            let v = Squares32x8::compute(c_vec, k);
            let mut tmp = [0u32; SQUARES32x8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, v);
            for j in 0..final_rem.len() {
                final_rem[j] = tmp[j] as f32 * SCALE;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn squares32x8_rand_i32s_chunk(
    chunk_idx: usize,
    chunk: &mut [i32],
    c0: u64,
    k: __m512i,
    range: u64,
    min: i32,
    lane_offsets: __m512i,
    k_step: __m512i,
    k_step2: __m512i,
    k_step3: __m512i,
    k_step4: __m512i,
) {
    unsafe {
        let c_start = c0.wrapping_add((chunk_idx * SQUARES32x8_PAR_CHUNK) as u64);
        let mut c_vec = _mm512_add_epi64(_mm512_set1_epi64(c_start as i64), lane_offsets);
        let vrange = _mm512_set1_epi64(range as i64);
        let vmin = _mm512_set1_epi32(min);

        let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
        let mut chunks32 = chunk.chunks_exact_mut(SQUARES32x8 * 4);

        for dst in chunks32.by_ref() {
            let v0 = Squares32x8::compute(c_vec, k);
            let v1 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step), k);
            let v2 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step2), k);
            let v3 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step3), k);

            #[inline(always)]
            unsafe fn pack_convert(
                v0: __m256i,
                v1: __m256i,
                vrange: __m512i,
                vmin: __m512i,
            ) -> __m512i {
                unsafe {
                    let l_u64 = _mm512_cvtepu32_epi64(v0);
                    let h_u64 = _mm512_cvtepu32_epi64(v1);

                    let res_l = _mm512_srli_epi64(_mm512_mul_epu32(l_u64, vrange), 32);
                    let res_h = _mm512_srli_epi64(_mm512_mul_epu32(h_u64, vrange), 32);

                    let packed_l = _mm512_cvtepi64_epi32(res_l);
                    let packed_h = _mm512_cvtepi64_epi32(res_h);

                    let res = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(packed_l), packed_h);
                    _mm512_add_epi32(res, vmin)
                }
            }

            let res01 = pack_convert(v0, v1, vrange, vmin);
            let res23 = pack_convert(v2, v3, vrange, vmin);

            if is_aligned {
                _mm512_stream_si512(dst.as_mut_ptr() as *mut _, res01);
                _mm512_stream_si512(dst[16..].as_mut_ptr() as *mut _, res23);
            } else {
                _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, res01);
                _mm512_storeu_si512(dst[16..].as_mut_ptr() as *mut _, res23);
            }

            c_vec = _mm512_add_epi64(c_vec, k_step4);
        }

        let rem = chunks32.into_remainder();
        let mut rem_chunks8 = rem.chunks_exact_mut(SQUARES32x8);
        for dst in rem_chunks8.by_ref() {
            let v = Squares32x8::compute(c_vec, k);
            let mut result = [0u32; SQUARES32x8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut _, v);
            for j in 0..SQUARES32x8 {
                dst[j] = ((result[j] as u64 * range) >> 32) as i32 + min;
            }
            c_vec = _mm512_add_epi64(c_vec, k_step);
        }
        let final_rem = rem_chunks8.into_remainder();
        if !final_rem.is_empty() {
            let v = Squares32x8::compute(c_vec, k);
            let mut tmp = [0u32; SQUARES32x8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, v);
            for j in 0..final_rem.len() {
                final_rem[j] = ((tmp[j] as u64 * range) >> 32) as i32 + min;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn squares32x8_rand_f32s_chunk(
    chunk_idx: usize,
    chunk: &mut [f32],
    c0: u64,
    k: __m512i,
    combined_scale: f32,
    min: f32,
    lane_offsets: __m512i,
    k_step: __m512i,
    k_step2: __m512i,
    k_step3: __m512i,
    k_step4: __m512i,
) {
    unsafe {
        let c_start = c0.wrapping_add((chunk_idx * SQUARES32x8_PAR_CHUNK) as u64);
        let mut c_vec = _mm512_add_epi64(_mm512_set1_epi64(c_start as i64), lane_offsets);
        let vscale = _mm512_set1_ps(combined_scale);
        let vmin = _mm512_set1_ps(min);

        let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
        let mut chunks32 = chunk.chunks_exact_mut(SQUARES32x8 * 4);

        for dst in chunks32.by_ref() {
            let v0 = Squares32x8::compute(c_vec, k);
            let v1 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step), k);
            let v2 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step2), k);
            let v3 = Squares32x8::compute(_mm512_add_epi64(c_vec, k_step3), k);

            let res01 = _mm512_cvtepu32_ps(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1));
            let res23 = _mm512_cvtepu32_ps(_mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3));

            let f01 = _mm512_add_ps(_mm512_mul_ps(res01, vscale), vmin);
            let f23 = _mm512_add_ps(_mm512_mul_ps(res23, vscale), vmin);

            if is_aligned {
                _mm512_stream_ps(dst.as_mut_ptr(), f01);
                _mm512_stream_ps(dst[16..].as_mut_ptr(), f23);
            } else {
                _mm512_storeu_ps(dst.as_mut_ptr(), f01);
                _mm512_storeu_ps(dst[16..].as_mut_ptr(), f23);
            }

            c_vec = _mm512_add_epi64(c_vec, k_step4);
        }

        let rem = chunks32.into_remainder();
        let mut rem_chunks8 = rem.chunks_exact_mut(SQUARES32x8);
        for dst in rem_chunks8.by_ref() {
            let v = Squares32x8::compute(c_vec, k);
            let mut result = [0u32; SQUARES32x8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut _, v);
            for j in 0..SQUARES32x8 {
                dst[j] = result[j] as f32 * combined_scale + min;
            }
            c_vec = _mm512_add_epi64(c_vec, k_step);
        }
        let final_rem = rem_chunks8.into_remainder();
        if !final_rem.is_empty() {
            let v = Squares32x8::compute(c_vec, k);
            let mut tmp = [0u32; SQUARES32x8];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, v);
            for j in 0..final_rem.len() {
                final_rem[j] = tmp[j] as f32 * combined_scale + min;
            }
        }
    }
}

/// Creates a new `Squares32` instance.
/// The caller is responsible for freeing the memory using `squares32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_new(seed: u32) -> *mut Squares32 {
    Box::into_raw(Box::new(Squares32::new(seed as u64)))
}

/// Frees the memory of a `Squares32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_free(ptr: *mut Squares32) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const SQUARES32_PAR_CHUNK: usize = 4096;

/// 4-way unrolled batch kernel for Squares32.
/// y0..y3 are independent lanes, each advanced by k4 = 4*k per batch.
/// Since z_i = y_i + k, and y_{i+1} = y_i + k, we get z_i == y_{i+1},
/// eliminating redundant adds. No loop-carried dependency within a batch.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_next_u32s(ptr: *mut Squares32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    // z_i = y_i + k == y1 = y0+k, z0 = y1, z1 = y2, z2 = y3
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1);
                    dst[1] = Squares32::compute_yz(y1, y2);
                    dst[2] = Squares32::compute_yz(y2, y3);
                    dst[3] = Squares32::compute_yz(y3, z3);
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr);
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// Uses a 4-way unrolled parallel kernel for high throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_next_f32s(ptr: *mut Squares32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let c0 = rng.c;
        let k = rng.k;
        let k4 = k.wrapping_mul(4);
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1) as f32 * SCALE;
                    dst[1] = Squares32::compute_yz(y1, y2) as f32 * SCALE;
                    dst[2] = Squares32::compute_yz(y2, y3) as f32 * SCALE;
                    dst[3] = Squares32::compute_yz(y3, z3) as f32 * SCALE;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr) as f32 * SCALE;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
/// Uses a 4-way unrolled parallel kernel for high throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_rand_i32s(
    ptr: *mut Squares32,
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
        let k4 = k.wrapping_mul(4);
        let range = (max as i64 - min as i64 + 1) as u64;

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = ((Squares32::compute_yz(y0, y1) as u64 * range) >> 32) as i32 + min;
                    dst[1] = ((Squares32::compute_yz(y1, y2) as u64 * range) >> 32) as i32 + min;
                    dst[2] = ((Squares32::compute_yz(y2, y3) as u64 * range) >> 32) as i32 + min;
                    dst[3] = ((Squares32::compute_yz(y3, z3) as u64 * range) >> 32) as i32 + min;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = ((Squares32::compute_yz(yr, zr) as u64 * range) >> 32) as i32 + min;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
/// Uses a 4-way unrolled parallel kernel for high throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32_rand_f32s(
    ptr: *mut Squares32,
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
        let k4 = k.wrapping_mul(4);
        let combined_scale = (max - min) * (1.0f32 / (u32::MAX as f32 + 1.0));

        buffer
            .par_chunks_mut(SQUARES32_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let c_start = c0.wrapping_add((chunk_idx * SQUARES32_PAR_CHUNK) as u64);
                let y_base = c_start.wrapping_mul(k);
                let mut y0 = y_base;
                let mut y1 = y_base.wrapping_add(k);
                let mut y2 = y1.wrapping_add(k);
                let mut y3 = y2.wrapping_add(k);

                let mut chunks4 = chunk.chunks_exact_mut(4);
                for dst in chunks4.by_ref() {
                    let z3 = y3.wrapping_add(k);
                    dst[0] = Squares32::compute_yz(y0, y1) as f32 * combined_scale + min;
                    dst[1] = Squares32::compute_yz(y1, y2) as f32 * combined_scale + min;
                    dst[2] = Squares32::compute_yz(y2, y3) as f32 * combined_scale + min;
                    dst[3] = Squares32::compute_yz(y3, z3) as f32 * combined_scale + min;
                    y0 = y0.wrapping_add(k4);
                    y1 = y1.wrapping_add(k4);
                    y2 = y2.wrapping_add(k4);
                    y3 = y3.wrapping_add(k4);
                }
                let rem = chunks4.into_remainder();
                let mut yr = y0;
                for dst in rem.iter_mut() {
                    let zr = yr.wrapping_add(k);
                    *dst = Squares32::compute_yz(yr, zr) as f32 * combined_scale + min;
                    yr = yr.wrapping_add(k);
                }
            });

        rng.c = rng.c.wrapping_add(count as u64);
    }
}

/// Creates a new `Squares32x8` instance.
/// The caller is responsible for freeing the memory using `squares32x8_free`.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_new(seed: u32) -> *mut Squares32x8 {
    unsafe { Box::into_raw(Box::new(Squares32x8::new(seed))) }
}

/// Frees the memory of a `Squares32x8` instance.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_free(ptr: *mut Squares32x8) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
/// This function utilizes AVX-512 SIMD and parallel processing for maximum throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_next_u32s(ptr: *mut Squares32x8, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut c_arr = [0u64; SQUARES32x8];
        _mm512_storeu_si512(c_arr.as_mut_ptr() as *mut _, rng.c);
        let c0 = c_arr[0];
        let k = rng.k;
        let lane_offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);

        buffer
            .par_chunks_mut(SQUARES32x8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                squares32x8_next_u32s_chunk(chunk_idx, chunk, c0, k, lane_offsets);
            });

        let num_generated = count as u64;
        for i in 0..SQUARES32x8 {
            c_arr[i] = c_arr[i].wrapping_add(num_generated);
        }
        rng.c = _mm512_loadu_si512(c_arr.as_ptr() as *const _);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// This function utilizes AVX-512 SIMD and parallel processing for maximum throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_next_f32s(ptr: *mut Squares32x8, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mut c_arr = [0u64; SQUARES32x8];
        _mm512_storeu_si512(c_arr.as_mut_ptr() as *mut _, rng.c);
        let c0 = c_arr[0];
        let k = rng.k;
        let lane_offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        let k_step = _mm512_set1_epi64(SQUARES32x8 as i64);
        let k_step2 = _mm512_set1_epi64((SQUARES32x8 * 2) as i64);
        let k_step3 = _mm512_set1_epi64((SQUARES32x8 * 3) as i64);
        let k_step4 = _mm512_set1_epi64((SQUARES32x8 * 4) as i64);

        buffer
            .par_chunks_mut(SQUARES32x8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                squares32x8_next_f32s_chunk(
                    chunk_idx,
                    chunk,
                    c0,
                    k,
                    lane_offsets,
                    k_step,
                    k_step2,
                    k_step3,
                    k_step4,
                );
            });

        let num_generated = count as u64;
        for i in 0..SQUARES32x8 {
            c_arr[i] = c_arr[i].wrapping_add(num_generated);
        }
        rng.c = _mm512_loadu_si512(c_arr.as_ptr() as *const _);
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
/// This function utilizes AVX-512 SIMD and parallel processing for maximum throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_rand_i32s(
    ptr: *mut Squares32x8,
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
        let mut c_arr = [0u64; SQUARES32x8];
        _mm512_storeu_si512(c_arr.as_mut_ptr() as *mut _, rng.c);
        let c0 = c_arr[0];
        let k = rng.k;
        let range = (max as i64 - min as i64 + 1) as u64;
        let lane_offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        let k_step = _mm512_set1_epi64(SQUARES32x8 as i64);
        let k_step2 = _mm512_set1_epi64((SQUARES32x8 * 2) as i64);
        let k_step3 = _mm512_set1_epi64((SQUARES32x8 * 3) as i64);
        let k_step4 = _mm512_set1_epi64((SQUARES32x8 * 4) as i64);

        buffer
            .par_chunks_mut(SQUARES32x8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                squares32x8_rand_i32s_chunk(
                    chunk_idx,
                    chunk,
                    c0,
                    k,
                    range,
                    min,
                    lane_offsets,
                    k_step,
                    k_step2,
                    k_step3,
                    k_step4,
                );
            });

        let num_generated = count as u64;
        for i in 0..SQUARES32x8 {
            c_arr[i] = c_arr[i].wrapping_add(num_generated);
        }
        rng.c = _mm512_loadu_si512(c_arr.as_ptr() as *const _);
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
/// This function utilizes AVX-512 SIMD and parallel processing for maximum throughput.
#[unsafe(no_mangle)]
pub extern "C" fn squares32x8_rand_f32s(
    ptr: *mut Squares32x8,
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
        let mut c_arr = [0u64; SQUARES32x8];
        _mm512_storeu_si512(c_arr.as_mut_ptr() as *mut _, rng.c);
        let c0 = c_arr[0];
        let k = rng.k;
        let combined_scale = (max - min) * (1.0f32 / (u32::MAX as f32 + 1.0));
        let lane_offsets = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        let k_step = _mm512_set1_epi64(SQUARES32x8 as i64);
        let k_step2 = _mm512_set1_epi64((SQUARES32x8 * 2) as i64);
        let k_step3 = _mm512_set1_epi64((SQUARES32x8 * 3) as i64);
        let k_step4 = _mm512_set1_epi64((SQUARES32x8 * 4) as i64);

        buffer
            .par_chunks_mut(SQUARES32x8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                squares32x8_rand_f32s_chunk(
                    chunk_idx,
                    chunk,
                    c0,
                    k,
                    combined_scale,
                    min,
                    lane_offsets,
                    k_step,
                    k_step2,
                    k_step3,
                    k_step4,
                );
            });

        let num_generated = count as u64;
        for i in 0..SQUARES32x8 {
            c_arr[i] = c_arr[i].wrapping_add(num_generated);
        }
        rng.c = _mm512_loadu_si512(c_arr.as_ptr() as *const _);
    }
}

/// Creates a new `Squares32Simd` instance, dispatching to AVX-512 or scalar implementation.
/// The caller is responsible for freeing the memory using `squares32simd_free`.
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_new(seed: u32) -> *mut Squares32Simd {
    dispatch_simd!(Squares32Simd, squares32_new, squares32x8_new, seed)
}
/// Frees the memory of a `Squares32Simd` instance.
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_free(ptr: *mut Squares32Simd) {
    dispatch_simd!(
        Squares32x8,
        Squares32,
        squares32_free,
        squares32x8_free,
        ptr
    )
}
/// Fills the output buffer with the next random `u32` values using the best available implementation.
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_next_u32s(ptr: *mut Squares32Simd, out: *mut u32, count: usize) {
    dispatch_simd!(
        Squares32x8,
        Squares32,
        squares32_next_u32s,
        squares32x8_next_u32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_next_f32s(ptr: *mut Squares32Simd, out: *mut f32, count: usize) {
    dispatch_simd!(
        Squares32x8,
        Squares32,
        squares32_next_f32s,
        squares32x8_next_f32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_rand_i32s(
    ptr: *mut Squares32Simd,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    dispatch_simd!(
        Squares32x8,
        Squares32,
        squares32_rand_i32s,
        squares32x8_rand_i32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn squares32simd_rand_f32s(
    ptr: *mut Squares32Simd,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    dispatch_simd!(
        Squares32x8,
        Squares32,
        squares32_rand_f32s,
        squares32x8_rand_f32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
