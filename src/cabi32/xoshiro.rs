use crate::rng::Rng32;
use crate::rng32::{Xoshiro128Pp, Xoshiro128Ppx16, Xoshiro128Ss, Xoshiro128Ssx16};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{ptr, slice::from_raw_parts_mut};

#[cfg(target_arch = "x86_64")]
const XOSHIRO128X16_LANES: usize = 16;
#[cfg(target_arch = "x86_64")]
const XOSHIRO128X16_PAR_CHUNK: usize = 1 << 20;

#[inline]
fn xoshiro_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
    let x = base_seed.wrapping_add((chunk_idx as u32).wrapping_mul(0x9E37_79B9));
    let mut z = x as u64;
    z ^= z >> 16;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 16;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    (z ^ (z >> 16)) as u32
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xoshiro128ppx16_base_seed(rng: &mut Xoshiro128Ppx16) -> u32 {
    let mut tmp = [0u32; XOSHIRO128X16_LANES];
    let v = unsafe { rng.nextu_vec() };
    unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v) };
    tmp[0]
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn xoshiro128ssx16_base_seed(rng: &mut Xoshiro128Ssx16) -> u32 {
    let mut tmp = [0u32; XOSHIRO128X16_LANES];
    let v = unsafe { rng.nextu_vec() };
    unsafe { _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v) };
    tmp[0]
}

// --- Xoshiro128++ ---

/// Creates a new `Xoshiro128Pp` instance.
/// The caller is responsible for freeing the memory using `xoshiro128pp_free`.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_new(seed: u32) -> *mut Xoshiro128Pp {
    Box::into_raw(Box::new(Xoshiro128Pp::new(seed)))
}

/// Frees the memory of a `Xoshiro128Pp` instance.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_free(ptr: *mut Xoshiro128Pp) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_next_u32s(ptr: *mut Xoshiro128Pp, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextu();
        }
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_next_f32s(ptr: *mut Xoshiro128Pp, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_rand_i32s(
    ptr: *mut Xoshiro128Pp,
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
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128pp_rand_f32s(
    ptr: *mut Xoshiro128Pp,
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
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

// --- Xoshiro128** ---

/// Creates a new `Xoshiro128Ss` instance.
/// The caller is responsible for freeing the memory using `xoshiro128ss_free`.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_new(seed: u32) -> *mut Xoshiro128Ss {
    Box::into_raw(Box::new(Xoshiro128Ss::new(seed)))
}

/// Frees the memory of a `Xoshiro128Ss` instance.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_free(ptr: *mut Xoshiro128Ss) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_next_u32s(ptr: *mut Xoshiro128Ss, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextu();
        }
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_next_f32s(ptr: *mut Xoshiro128Ss, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_rand_i32s(
    ptr: *mut Xoshiro128Ss,
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
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ss_rand_f32s(
    ptr: *mut Xoshiro128Ss,
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
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

// --- Xoshiro128++ x16 ---

/// Creates a new `Xoshiro128Ppx16` instance.
/// The caller is responsible for freeing the memory using `xoshiro128ppx16_free`.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_new(seed: u32) -> *mut Xoshiro128Ppx16 {
    unsafe { Box::into_raw(Box::new(Xoshiro128Ppx16::new(seed))) }
}

/// Frees the memory of a `Xoshiro128Ppx16` instance.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_free(ptr: *mut Xoshiro128Ppx16) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ppx16_next_u32s_chunk(rng: &mut Xoshiro128Ppx16, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = XOSHIRO128X16_LANES * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0u32; XOSHIRO128X16_LANES];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ppx16_next_f32s_chunk(
    rng: &mut Xoshiro128Ppx16,
    chunk: &mut [f32],
    scale: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; XOSHIRO128X16_LANES];
        let v = rng.nextfv(scale);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ppx16_rand_i32s_chunk(
    rng: &mut Xoshiro128Ppx16,
    chunk: &mut [i32],
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0i32; XOSHIRO128X16_LANES];
        let v = rng.randi_vec(v_range, v_min);
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ppx16_rand_f32s_chunk(
    rng: &mut Xoshiro128Ppx16,
    chunk: &mut [f32],
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; XOSHIRO128X16_LANES];
        let v = rng.randf_vec(v_mult, v_min);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_next_u32s(
    ptr: *mut Xoshiro128Ppx16,
    out: *mut u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = xoshiro128ppx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ppx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                xoshiro128ppx16_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_next_f32s(
    ptr: *mut Xoshiro128Ppx16,
    out: *mut f32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = xoshiro128ppx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ppx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
                xoshiro128ppx16_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_rand_i32s(
    ptr: *mut Xoshiro128Ppx16,
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
        let base_seed = xoshiro128ppx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ppx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
                let v_min = _mm512_set1_epi32(min);
                xoshiro128ppx16_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ppx16_rand_f32s(
    ptr: *mut Xoshiro128Ppx16,
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
        let base_seed = xoshiro128ppx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ppx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let v_mult = _mm512_set1_ps((max - min) * (1.0 / (u32::MAX as f32 + 1.0)));
                let v_min = _mm512_set1_ps(min);
                xoshiro128ppx16_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}

// --- Xoshiro128** x16 ---

/// Creates a new `Xoshiro128Ssx16` instance.
/// The caller is responsible for freeing the memory using `xoshiro128ssx16_free`.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_new(seed: u32) -> *mut Xoshiro128Ssx16 {
    unsafe { Box::into_raw(Box::new(Xoshiro128Ssx16::new(seed))) }
}

/// Frees the memory of a `Xoshiro128Ssx16` instance.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_free(ptr: *mut Xoshiro128Ssx16) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ssx16_next_u32s_chunk(rng: &mut Xoshiro128Ssx16, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;
    const UNROLL: usize = XOSHIRO128X16_LANES * 4;

    if aligned {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(XOSHIRO128X16_LANES * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextu_vec();
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= UNROLL {
            let v0 = rng.nextu_vec();
            let v1 = rng.nextu_vec();
            let v2 = rng.nextu_vec();
            let v3 = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(XOSHIRO128X16_LANES * 3) as *mut _, v3);
            out_ptr = out_ptr.add(UNROLL);
            remaining -= UNROLL;
        }
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextu_vec();
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0u32; XOSHIRO128X16_LANES];
        let v = rng.nextu_vec();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ssx16_next_f32s_chunk(
    rng: &mut Xoshiro128Ssx16,
    chunk: &mut [f32],
    scale: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; XOSHIRO128X16_LANES];
        let v = rng.nextfv(scale);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ssx16_rand_i32s_chunk(
    rng: &mut Xoshiro128Ssx16,
    chunk: &mut [i32],
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randi_vec(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0i32; XOSHIRO128X16_LANES];
        let v = rng.randi_vec(v_range, v_min);
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoshiro128ssx16_rand_f32s_chunk(
    rng: &mut Xoshiro128Ssx16,
    chunk: &mut [f32],
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    } else {
        while remaining >= XOSHIRO128X16_LANES {
            let v = rng.randf_vec(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(XOSHIRO128X16_LANES);
            remaining -= XOSHIRO128X16_LANES;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; XOSHIRO128X16_LANES];
        let v = rng.randf_vec(v_mult, v_min);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_next_u32s(
    ptr: *mut Xoshiro128Ssx16,
    out: *mut u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = xoshiro128ssx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ssx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                xoshiro128ssx16_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_next_f32s(
    ptr: *mut Xoshiro128Ssx16,
    out: *mut f32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = xoshiro128ssx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ssx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let scale = _mm512_set1_ps(1.0 / (u32::MAX as f32 + 1.0));
                xoshiro128ssx16_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_rand_i32s(
    ptr: *mut Xoshiro128Ssx16,
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
        let base_seed = xoshiro128ssx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ssx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
                let v_min = _mm512_set1_epi32(min);
                xoshiro128ssx16_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro128ssx16_rand_f32s(
    ptr: *mut Xoshiro128Ssx16,
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
        let base_seed = xoshiro128ssx16_base_seed(rng);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(XOSHIRO128X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoshiro128Ssx16::new(xoshiro_chunk_seed(base_seed, chunk_idx));
                let v_mult = _mm512_set1_ps((max - min) * (1.0 / (u32::MAX as f32 + 1.0)));
                let v_min = _mm512_set1_ps(min);
                xoshiro128ssx16_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}
