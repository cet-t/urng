#[cfg(target_arch = "x86_64")]
use crate::rng32::jsf::JSF32X16;
use crate::{
    rng::Rng32,
    rng32::{Jsf32, jsf::Jsf32x16},
};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{ptr, slice::from_raw_parts_mut};

/// Creates a new `Jsf32` instance.
/// The caller is responsible for freeing the memory using `jsf32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_new(seed: u32) -> *mut Jsf32 {
    Box::into_raw(Box::new(Jsf32::new(seed)))
}

/// Frees the memory of a `Jsf32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_free(ptr: *mut Jsf32) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

const JSF32_PAR_CHUNK: usize = 0x1000;

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_next_u32s(ptr: *mut Jsf32, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Jsf32::new(seed.wrapping_add(i as u32));
                let mut chunks4 = chunk.chunks_exact_mut(4);
                for c in chunks4.by_ref() {
                    c[0] = local_rng.nextu();
                    c[1] = local_rng.nextu();
                    c[2] = local_rng.nextu();
                    c[3] = local_rng.nextu();
                }
                for x in chunks4.into_remainder() {
                    *x = local_rng.nextu();
                }
            });
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_next_f32s(ptr: *mut Jsf32, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Jsf32::new(base_seed.wrapping_add(i as u32));
                let mut chunks4 = chunk.chunks_exact_mut(4);
                for c in chunks4.by_ref() {
                    c[0] = local_rng.nextf();
                    c[1] = local_rng.nextf();
                    c[2] = local_rng.nextf();
                    c[3] = local_rng.nextf();
                }
                for x in chunks4.into_remainder() {
                    *x = local_rng.nextf();
                }
            });
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_rand_i32s(
    ptr: *mut Jsf32,
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
        let base_seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Jsf32::new(base_seed.wrapping_add(i as u32));
                let mut chunks4 = chunk.chunks_exact_mut(4);
                for c in chunks4.by_ref() {
                    c[0] = local_rng.randi(min, max);
                    c[1] = local_rng.randi(min, max);
                    c[2] = local_rng.randi(min, max);
                    c[3] = local_rng.randi(min, max);
                }
                for x in chunks4.into_remainder() {
                    *x = local_rng.randi(min, max);
                }
            });
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn jsf32_rand_f32s(
    ptr: *mut Jsf32,
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
        let base_seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Jsf32::new(base_seed.wrapping_add(i as u32));
                let mut chunks4 = chunk.chunks_exact_mut(4);
                for c in chunks4.by_ref() {
                    c[0] = local_rng.randf(min, max);
                    c[1] = local_rng.randf(min, max);
                    c[2] = local_rng.randf(min, max);
                    c[3] = local_rng.randf(min, max);
                }
                for x in chunks4.into_remainder() {
                    *x = local_rng.randf(min, max);
                }
            });
    }
}

// --- Jsf32x16 (AVX-512) ---

#[cfg(target_arch = "x86_64")]
const JSF32X16_PAR_CHUNK: usize = 1 << 20;

#[cfg(target_arch = "x86_64")]
#[inline]
fn jsf32x16_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
    let x = base_seed.wrapping_add((chunk_idx as u32).wrapping_mul(0x9E37_79B9));
    let mut z = x as u64;
    z ^= z >> 16;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 16;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    (z ^ (z >> 16)) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn jsf32x16_next_u32s_chunk(rng: &mut Jsf32x16, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = JSF32X16 * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(JSF32X16) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(JSF32X16 * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(JSF32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(JSF32X16) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(JSF32X16 * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(JSF32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0u32; JSF32X16];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn jsf32x16_next_f32s_chunk(rng: &mut Jsf32x16, chunk: &mut [f32], scale: __m512) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = JSF32X16 * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.nextfv(scale);
            let v1 = rng.nextfv(scale);
            let v2 = rng.nextfv(scale);
            let v3 = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v0);
            _mm512_stream_ps(out_ptr.add(JSF32X16), v1);
            _mm512_stream_ps(out_ptr.add(JSF32X16 * 2), v2);
            _mm512_stream_ps(out_ptr.add(JSF32X16 * 3), v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.nextfv(scale);
            let v1 = rng.nextfv(scale);
            let v2 = rng.nextfv(scale);
            let v3 = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v0);
            _mm512_storeu_ps(out_ptr.add(JSF32X16), v1);
            _mm512_storeu_ps(out_ptr.add(JSF32X16 * 2), v2);
            _mm512_storeu_ps(out_ptr.add(JSF32X16 * 3), v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; JSF32X16];
        let v = rng.nextfv(scale);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn jsf32x16_rand_i32s_chunk(
    rng: &mut Jsf32x16,
    chunk: &mut [i32],
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = JSF32X16 * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.randi_vec(v_range, v_min);
            let v1 = rng.randi_vec(v_range, v_min);
            let v2 = rng.randi_vec(v_range, v_min);
            let v3 = rng.randi_vec(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(JSF32X16) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(JSF32X16 * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(JSF32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.randi_vec(v_range, v_min);
            let v1 = rng.randi_vec(v_range, v_min);
            let v2 = rng.randi_vec(v_range, v_min);
            let v3 = rng.randi_vec(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(JSF32X16) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(JSF32X16 * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(JSF32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0i32; JSF32X16];
        let v = rng.randi_vec(v_range, v_min);
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn jsf32x16_rand_f32s_chunk(
    rng: &mut Jsf32x16,
    chunk: &mut [f32],
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = JSF32X16 * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.randf_vec(v_mult, v_min);
            let v1 = rng.randf_vec(v_mult, v_min);
            let v2 = rng.randf_vec(v_mult, v_min);
            let v3 = rng.randf_vec(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v0);
            _mm512_stream_ps(out_ptr.add(JSF32X16), v1);
            _mm512_stream_ps(out_ptr.add(JSF32X16 * 2), v2);
            _mm512_stream_ps(out_ptr.add(JSF32X16 * 3), v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.randf_vec(v_mult, v_min);
            let v1 = rng.randf_vec(v_mult, v_min);
            let v2 = rng.randf_vec(v_mult, v_min);
            let v3 = rng.randf_vec(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v0);
            _mm512_storeu_ps(out_ptr.add(JSF32X16), v1);
            _mm512_storeu_ps(out_ptr.add(JSF32X16 * 2), v2);
            _mm512_storeu_ps(out_ptr.add(JSF32X16 * 3), v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= JSF32X16 {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(JSF32X16);
            remaining -= JSF32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; JSF32X16];
        let v = rng.randf_vec(v_mult, v_min);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

/// Creates a new `Jsf32x16` instance.
/// The caller is responsible for freeing the memory using `jsf32x16_free`.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_new(seed: u32) -> *mut Jsf32x16 {
    unsafe { Box::into_raw(Box::new(Jsf32x16::new(seed))) }
}

/// Frees the memory of a `Jsf32x16` instance.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_free(ptr: *mut Jsf32x16) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_next_u32s(ptr: *mut Jsf32x16, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; JSF32X16];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Jsf32x16::new(jsf32x16_chunk_seed(base_seed, chunk_idx));
                jsf32x16_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_next_f32s(ptr: *mut Jsf32x16, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; JSF32X16];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Jsf32x16::new(jsf32x16_chunk_seed(base_seed, chunk_idx));
                let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
                jsf32x16_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_rand_i32s(
    ptr: *mut Jsf32x16,
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
        let mut tmp = [0u32; JSF32X16];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Jsf32x16::new(jsf32x16_chunk_seed(base_seed, chunk_idx));
                let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
                let v_min = _mm512_set1_epi32(min);
                jsf32x16_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn jsf32x16_rand_f32s(
    ptr: *mut Jsf32x16,
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
        let mut tmp = [0u32; JSF32X16];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(JSF32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Jsf32x16::new(jsf32x16_chunk_seed(base_seed, chunk_idx));
                let v_mult = _mm512_set1_ps((max - min) * (1.0 / (u32::MAX as f32 + 1.0)));
                let v_min = _mm512_set1_ps(min);
                jsf32x16_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}
