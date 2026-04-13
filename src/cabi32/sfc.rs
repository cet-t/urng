use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{ptr, slice::from_raw_parts_mut};

use crate::{
    _internal::FSCALE32,
    rng::Rng32,
    rng32::{
        Sfc32,
        sfc::{SFC32X16, Sfc32x16},
    },
};

// --- Sfc32 ---

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_new(seed: u32) -> *mut Sfc32 {
    Box::into_raw(Box::new(Sfc32::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_free(ptr: *mut Sfc32) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_next_u32s(ptr: *mut Sfc32, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextu();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_next_f32s(ptr: *mut Sfc32, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_rand_i32s(
    ptr: *mut Sfc32,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32_rand_f32s(
    ptr: *mut Sfc32,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

// --- Sfc32x16 ---

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_new(seed: u32) -> *mut Sfc32x16 {
    unsafe { Box::into_raw(Box::new(Sfc32x16::new(seed))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_free(ptr: *mut Sfc32x16) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
const SFC32X16_PAR_CHUNK: usize = 0x20000;
#[cfg(target_arch = "x86_64")]
const SFC32X16_UNROLL: usize = SFC32X16 << 2;

#[cfg(target_arch = "x86_64")]
#[inline]
fn sfc32x16_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
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
unsafe fn sfc32x16_next_u32s_chunk(rng: &mut Sfc32x16, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.nextuv();
            let v1 = rng.nextuv();
            let v2 = rng.nextuv();
            let v3 = rng.nextuv();
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(SFC32X16) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(SFC32X16 * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(SFC32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextuv();
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    } else {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.nextuv();
            let v1 = rng.nextuv();
            let v2 = rng.nextuv();
            let v3 = rng.nextuv();
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(SFC32X16) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(SFC32X16 * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(SFC32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextuv();
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0u32; SFC32X16];
        let v = rng.nextuv();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x16_next_f32s_chunk(rng: &mut Sfc32x16, chunk: &mut [f32], scale: __m512) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.nextfv_scaled(scale);
            let v1 = rng.nextfv_scaled(scale);
            let v2 = rng.nextfv_scaled(scale);
            let v3 = rng.nextfv_scaled(scale);
            _mm512_stream_ps(out_ptr, v0);
            _mm512_stream_ps(out_ptr.add(SFC32X16), v1);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextfv_scaled(scale);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    } else {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.nextfv_scaled(scale);
            let v1 = rng.nextfv_scaled(scale);
            let v2 = rng.nextfv_scaled(scale);
            let v3 = rng.nextfv_scaled(scale);
            _mm512_storeu_ps(out_ptr, v0);
            _mm512_storeu_ps(out_ptr.add(SFC32X16), v1);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextfv_scaled(scale);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; SFC32X16];
        let v = rng.nextfv_scaled(scale);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x16_rand_i32s_chunk(
    rng: &mut Sfc32x16,
    chunk: &mut [i32],
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.randiv(v_range, v_min);
            let v1 = rng.randiv(v_range, v_min);
            let v2 = rng.randiv(v_range, v_min);
            let v3 = rng.randiv(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v0);
            _mm512_stream_si512(out_ptr.add(SFC32X16) as *mut _, v1);
            _mm512_stream_si512(out_ptr.add(SFC32X16 * 2) as *mut _, v2);
            _mm512_stream_si512(out_ptr.add(SFC32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.randiv(v_range, v_min);
            _mm512_stream_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    } else {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.randiv(v_range, v_min);
            let v1 = rng.randiv(v_range, v_min);
            let v2 = rng.randiv(v_range, v_min);
            let v3 = rng.randiv(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v0);
            _mm512_storeu_si512(out_ptr.add(SFC32X16) as *mut _, v1);
            _mm512_storeu_si512(out_ptr.add(SFC32X16 * 2) as *mut _, v2);
            _mm512_storeu_si512(out_ptr.add(SFC32X16 * 3) as *mut _, v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.randiv(v_range, v_min);
            _mm512_storeu_si512(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0i32; SFC32X16];
        let v = rng.randiv(v_range, v_min);
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x16_rand_f32s_chunk(
    rng: &mut Sfc32x16,
    chunk: &mut [f32],
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.randfv(v_mult, v_min);
            let v1 = rng.randfv(v_mult, v_min);
            let v2 = rng.randfv(v_mult, v_min);
            let v3 = rng.randfv(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v0);
            _mm512_stream_ps(out_ptr.add(SFC32X16), v1);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.randfv(v_mult, v_min);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    } else {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.randfv(v_mult, v_min);
            let v1 = rng.randfv(v_mult, v_min);
            let v2 = rng.randfv(v_mult, v_min);
            let v3 = rng.randfv(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v0);
            _mm512_storeu_ps(out_ptr.add(SFC32X16), v1);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.randfv(v_mult, v_min);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; SFC32X16];
        let v = rng.randfv(v_mult, v_min);
        _mm512_storeu_ps(tmp.as_mut_ptr(), v);
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_next_u32s(ptr: *mut Sfc32x16, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; SFC32X16];
        let v = rng.nextuv();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x16::new(sfc32x16_chunk_seed(base_seed, chunk_idx));
                sfc32x16_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_next_f32s(ptr: *mut Sfc32x16, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; SFC32X16];
        let v = rng.nextuv();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let scale = _mm512_set1_ps(FSCALE32);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x16::new(sfc32x16_chunk_seed(base_seed, chunk_idx));
                sfc32x16_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_rand_i32s(
    ptr: *mut Sfc32x16,
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
        let mut tmp = [0u32; SFC32X16];
        let v = rng.nextuv();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let v_min = _mm512_set1_epi32(min);
        let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x16::new(sfc32x16_chunk_seed(base_seed, chunk_idx));
                sfc32x16_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x16_rand_f32s(
    ptr: *mut Sfc32x16,
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
        let mut tmp = [0u32; SFC32X16];
        let v = rng.nextuv();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        let base_seed = tmp[0];

        let v_mult = _mm512_set1_ps((max - min) * FSCALE32);
        let v_min = _mm512_set1_ps(min);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x16::new(sfc32x16_chunk_seed(base_seed, chunk_idx));
                sfc32x16_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}
