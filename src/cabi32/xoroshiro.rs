#[cfg(target_arch = "x86_64")]
use crate::rng32::xoroshiro::{XOROSHIRO64SSX8, XOROSHIRO64SSX16};
use crate::{
    _internal::{chunk_seed32, fill_chunk_auto, prefer_nt},
    rng::Rng32,
    rng32::{
        Xoroshiro64Ss,
        xoroshiro::{Xoroshiro64Ssx8, Xoroshiro64Ssx16},
    },
};
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{ptr, slice::from_raw_parts_mut};

// --- Xoroshiro64Ss (scalar) ---

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_new(seed: u32) -> *mut Xoroshiro64Ss {
    Box::into_raw(Box::new(Xoroshiro64Ss::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_free(ptr: *mut Xoroshiro64Ss) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOROSHIRO64SS_PAR_CHUNK: usize = 0x20000;

/// Fills `buffer` in parallel: each chunk runs its own decorrelated
/// `Xoroshiro64Ss`, producing one element per `step` call. Sixteen
/// outputs (64 bytes for 4-byte `T`) are batched per generator call so
/// the non-temporal path can stream whole cache lines.
#[inline(always)]
fn xoro64ss_fill<T, M>(buffer: &mut [T], seed: u32, step: M)
where
    T: Copy + Default + Send,
    M: Fn(&mut Xoroshiro64Ss) -> T + Sync,
{
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(XOROSHIRO64SS_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut local_rng = Xoroshiro64Ss::new(seed.wrapping_add(chunk_idx as u32));
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 16];
                    for v in &mut out {
                        *v = step(&mut local_rng);
                    }
                    out
                });
            }
        });
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_next_u32s(ptr: *mut Xoroshiro64Ss, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        xoro64ss_fill(buffer, seed, |r| r.nextu());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_next_f32s(ptr: *mut Xoroshiro64Ss, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        xoro64ss_fill(buffer, seed, |r| r.nextf());
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_rand_i32s(
    ptr: *mut Xoroshiro64Ss,
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
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        xoro64ss_fill(buffer, seed, |r| r.randi(min, max));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_rand_f32s(
    ptr: *mut Xoroshiro64Ss,
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
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        xoro64ss_fill(buffer, seed, |r| r.randf(min, max));
    }
}

// --- Xoroshiro64Ssx8 (AVX2) ---

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_new(seed: u32) -> *mut Xoroshiro64Ssx8 {
    unsafe { Box::into_raw(Box::new(Xoroshiro64Ssx8::new(seed))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_free(ptr: *mut Xoroshiro64Ssx8) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
const XOROSHIRO64SSX8_PAR_CHUNK: usize = 0x20000;
#[cfg(target_arch = "x86_64")]
const XOROSHIRO64SSX8_UNROLL: usize = XOROSHIRO64SSX8 << 2;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx8_next_u32s_chunk(rng: &mut Xoroshiro64Ssx8, chunk: &mut [u32], nt: bool) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_si256(out_ptr as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_stream_si256(out_ptr as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    } else {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_storeu_si256(out_ptr as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_storeu_si256(out_ptr as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    }
    if remaining > 0 {
        let mut tmp = [0u32; XOROSHIRO64SSX8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx8_next_f32s_chunk(
    rng: &mut Xoroshiro64Ssx8,
    chunk: &mut [f32],
    nt: bool,
    scale: __m256,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_ps(out_ptr, rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 2), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_stream_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    } else {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_storeu_ps(out_ptr, rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 2), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_storeu_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    }
    if remaining > 0 {
        let mut tmp = [0f32; XOROSHIRO64SSX8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), rng.nextfv(scale));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx8_rand_i32s_chunk(
    rng: &mut Xoroshiro64Ssx8,
    chunk: &mut [i32],
    nt: bool,
    v_range: __m256i,
    v_min: __m256i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            _mm256_stream_si256(
                out_ptr.add(XOROSHIRO64SSX8) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_stream_si256(
                out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_stream_si256(
                out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_stream_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    } else {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_storeu_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            _mm256_storeu_si256(
                out_ptr.add(XOROSHIRO64SSX8) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_storeu_si256(
                out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_storeu_si256(
                out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_storeu_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    }
    if remaining > 0 {
        let mut tmp = [0i32; XOROSHIRO64SSX8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx8_rand_f32s_chunk(
    rng: &mut Xoroshiro64Ssx8,
    chunk: &mut [f32],
    nt: bool,
    v_mult: __m256,
    v_min: __m256,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_ps(out_ptr, rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 2), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_stream_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    } else {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_storeu_ps(out_ptr, rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 2), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8_UNROLL);
            remaining -= XOROSHIRO64SSX8_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX8 {
            _mm256_storeu_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX8);
            remaining -= XOROSHIRO64SSX8;
        }
    }
    if remaining > 0 {
        let mut tmp = [0f32; XOROSHIRO64SSX8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_next_u32s(
    ptr: *mut Xoroshiro64Ssx8,
    out: *mut u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed: u32 = rng.nextu()[0];
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx8::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx8_next_u32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_next_f32s(
    ptr: *mut Xoroshiro64Ssx8,
    out: *mut f32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed: u32 = rng.nextu()[0];
        let scale = _mm256_set1_ps(crate::_internal::FSCALE32);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx8::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx8_next_f32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    scale,
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_rand_i32s(
    ptr: *mut Xoroshiro64Ssx8,
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
        let base_seed: u32 = rng.nextu()[0];
        let v_range = _mm256_set1_epi64x(max as i64 - min as i64 + 1);
        let v_min = _mm256_set1_epi32(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx8::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx8_rand_i32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    v_range,
                    v_min,
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_rand_f32s(
    ptr: *mut Xoroshiro64Ssx8,
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
        let base_seed: u32 = rng.nextu()[0];
        let v_mult = _mm256_set1_ps((max - min) * crate::_internal::FSCALE32);
        let v_min = _mm256_set1_ps(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx8::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx8_rand_f32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    v_mult,
                    v_min,
                );
            });
    }
}

// --- Xoroshiro64Ssx16 (AVX-512) ---

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_new(seed: u32) -> *mut Xoroshiro64Ssx16 {
    unsafe { Box::into_raw(Box::new(Xoroshiro64Ssx16::new(seed))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_free(ptr: *mut Xoroshiro64Ssx16) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
const XOROSHIRO64SSX16_PAR_CHUNK: usize = 1 << 20;
#[cfg(target_arch = "x86_64")]
const XOROSHIRO64SSX16_UNROLL: usize = XOROSHIRO64SSX16 * 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx16_next_u32s_chunk(
    rng: &mut Xoroshiro64Ssx16,
    chunk: &mut [u32],
    nt: bool,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_si512(out_ptr as *mut _, rng.nextuv());
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16) as *mut _, rng.nextuv());
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _, rng.nextuv());
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_stream_si512(out_ptr as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    } else {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_storeu_si512(out_ptr as *mut _, rng.nextuv());
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16) as *mut _, rng.nextuv());
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _, rng.nextuv());
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_storeu_si512(out_ptr as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    }
    if remaining > 0 {
        let mut tmp = [0u32; XOROSHIRO64SSX16];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx16_next_f32s_chunk(
    rng: &mut Xoroshiro64Ssx16,
    chunk: &mut [f32],
    nt: bool,
    scale: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_ps(out_ptr, rng.nextfv(scale));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16), rng.nextfv(scale));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 2), rng.nextfv(scale));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_stream_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    } else {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_storeu_ps(out_ptr, rng.nextfv(scale));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16), rng.nextfv(scale));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 2), rng.nextfv(scale));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_storeu_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    }
    if remaining > 0 {
        let mut tmp = [0f32; XOROSHIRO64SSX16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), rng.nextfv(scale));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx16_rand_i32s_chunk(
    rng: &mut Xoroshiro64Ssx16,
    chunk: &mut [i32],
    nt: bool,
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_si512(out_ptr as *mut _, rng.randiv(v_range, v_min));
            _mm512_stream_si512(
                out_ptr.add(XOROSHIRO64SSX16) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm512_stream_si512(
                out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm512_stream_si512(
                out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_stream_si512(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    } else {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_storeu_si512(out_ptr as *mut _, rng.randiv(v_range, v_min));
            _mm512_storeu_si512(
                out_ptr.add(XOROSHIRO64SSX16) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm512_storeu_si512(
                out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm512_storeu_si512(
                out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_storeu_si512(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    }
    if remaining > 0 {
        let mut tmp = [0i32; XOROSHIRO64SSX16];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx16_rand_f32s_chunk(
    rng: &mut Xoroshiro64Ssx16,
    chunk: &mut [f32],
    nt: bool,
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = nt && (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_ps(out_ptr, rng.randfv(v_mult, v_min));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16), rng.randfv(v_mult, v_min));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 2), rng.randfv(v_mult, v_min));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_stream_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    } else {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_storeu_ps(out_ptr, rng.randfv(v_mult, v_min));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16), rng.randfv(v_mult, v_min));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 2), rng.randfv(v_mult, v_min));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16_UNROLL);
            remaining -= XOROSHIRO64SSX16_UNROLL;
        }
        while remaining >= XOROSHIRO64SSX16 {
            _mm512_storeu_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(XOROSHIRO64SSX16);
            remaining -= XOROSHIRO64SSX16;
        }
    }
    if remaining > 0 {
        let mut tmp = [0f32; XOROSHIRO64SSX16];
        _mm512_storeu_ps(tmp.as_mut_ptr(), rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_next_u32s(
    ptr: *mut Xoroshiro64Ssx16,
    out: *mut u32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed: u32 = rng.nextu()[0];
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx16::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx16_next_u32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_next_f32s(
    ptr: *mut Xoroshiro64Ssx16,
    out: *mut f32,
    count: usize,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed: u32 = rng.nextu()[0];
        let scale = _mm512_set1_ps(crate::_internal::FSCALE32);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx16::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx16_next_f32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    scale,
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_rand_i32s(
    ptr: *mut Xoroshiro64Ssx16,
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
        let base_seed: u32 = rng.nextu()[0];
        let v_range = _mm512_set1_epi64(max as i64 - min as i64 + 1);
        let v_min = _mm512_set1_epi32(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx16::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx16_rand_i32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    v_range,
                    v_min,
                );
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_rand_f32s(
    ptr: *mut Xoroshiro64Ssx16,
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
        let base_seed: u32 = rng.nextu()[0];
        let v_mult = _mm512_set1_ps((max - min) * crate::_internal::FSCALE32);
        let v_min = _mm512_set1_ps(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx16::new(chunk_seed32(base_seed, chunk_idx));
                xoroshiro64ssx16_rand_f32s_chunk(
                    &mut local_rng,
                    chunk,
                    crate::_internal::prefer_nt_for(count, chunk),
                    v_mult,
                    v_min,
                );
            });
    }
}
