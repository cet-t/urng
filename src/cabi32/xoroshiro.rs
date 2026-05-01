#[cfg(target_arch = "x86_64")]
use crate::rng32::xoroshiro::{XOROSHIRO64SSX8, XOROSHIRO64SSX16};
use crate::{
    rng::Rng32,
    rng32::{Xoroshiro64Ss, xoroshiro::{Xoroshiro64Ssx8, Xoroshiro64Ssx16}},
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

const XOROSHIRO64SS_PAR_CHUNK: usize = 0x1000;

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_next_u32s(ptr: *mut Xoroshiro64Ss, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Xoroshiro64Ss::new(seed.wrapping_add(i as u32));
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

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ss_next_f32s(ptr: *mut Xoroshiro64Ss, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Xoroshiro64Ss::new(seed.wrapping_add(i as u32));
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
        buffer
            .par_chunks_mut(XOROSHIRO64SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Xoroshiro64Ss::new(seed.wrapping_add(i as u32));
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
        buffer
            .par_chunks_mut(XOROSHIRO64SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(i, chunk)| {
                let mut local_rng = Xoroshiro64Ss::new(seed.wrapping_add(i as u32));
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
fn xoroshiro64ssx8_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
    let x = base_seed.wrapping_add((chunk_idx as u32).wrapping_mul(0x9E37_79B9));
    let mut z = x as u64;
    z ^= z >> 16;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 16;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    (z ^ (z >> 16)) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xoroshiro64ssx8_next_u32s_chunk(rng: &mut Xoroshiro64Ssx8, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 0) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 1) as *mut _, rng.nextuv());
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
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 0) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 1) as *mut _, rng.nextuv());
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
unsafe fn xoroshiro64ssx8_next_f32s_chunk(rng: &mut Xoroshiro64Ssx8, chunk: &mut [f32], scale: __m256) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 0), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 1), rng.nextfv(scale));
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
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 0), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 1), rng.nextfv(scale));
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
    v_range: __m256i,
    v_min: __m256i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 0) as *mut _, rng.randiv(v_range, v_min));
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 1) as *mut _, rng.randiv(v_range, v_min));
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _, rng.randiv(v_range, v_min));
            _mm256_stream_si256(out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _, rng.randiv(v_range, v_min));
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
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 0) as *mut _, rng.randiv(v_range, v_min));
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 1) as *mut _, rng.randiv(v_range, v_min));
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 2) as *mut _, rng.randiv(v_range, v_min));
            _mm256_storeu_si256(out_ptr.add(XOROSHIRO64SSX8 * 3) as *mut _, rng.randiv(v_range, v_min));
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
    v_mult: __m256,
    v_min: __m256,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX8_UNROLL {
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 0), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(XOROSHIRO64SSX8 * 1), rng.randfv(v_mult, v_min));
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
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 0), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(XOROSHIRO64SSX8 * 1), rng.randfv(v_mult, v_min));
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
pub extern "C" fn xoroshiro64ssx8_next_u32s(ptr: *mut Xoroshiro64Ssx8, out: *mut u32, count: usize) {
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
                let mut local_rng = Xoroshiro64Ssx8::new(xoroshiro64ssx8_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx8_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx8_next_f32s(ptr: *mut Xoroshiro64Ssx8, out: *mut f32, count: usize) {
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
                let mut local_rng = Xoroshiro64Ssx8::new(xoroshiro64ssx8_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx8_next_f32s_chunk(&mut local_rng, chunk, scale);
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
        let v_range = _mm256_set1_epi64x((max as i64 - min as i64 + 1) as i64);
        let v_min = _mm256_set1_epi32(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx8::new(xoroshiro64ssx8_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx8_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
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
                let mut local_rng = Xoroshiro64Ssx8::new(xoroshiro64ssx8_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx8_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
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
fn xoroshiro64ssx16_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
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
unsafe fn xoroshiro64ssx16_next_u32s_chunk(rng: &mut Xoroshiro64Ssx16, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 0) as *mut _, rng.nextuv());
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 1) as *mut _, rng.nextuv());
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
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 0) as *mut _, rng.nextuv());
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 1) as *mut _, rng.nextuv());
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
unsafe fn xoroshiro64ssx16_next_f32s_chunk(rng: &mut Xoroshiro64Ssx16, chunk: &mut [f32], scale: __m512) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 0), rng.nextfv(scale));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 1), rng.nextfv(scale));
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
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 0), rng.nextfv(scale));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 1), rng.nextfv(scale));
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
    v_range: __m512i,
    v_min: __m512i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 0) as *mut _, rng.randiv(v_range, v_min));
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 1) as *mut _, rng.randiv(v_range, v_min));
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _, rng.randiv(v_range, v_min));
            _mm512_stream_si512(out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _, rng.randiv(v_range, v_min));
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
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 0) as *mut _, rng.randiv(v_range, v_min));
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 1) as *mut _, rng.randiv(v_range, v_min));
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 2) as *mut _, rng.randiv(v_range, v_min));
            _mm512_storeu_si512(out_ptr.add(XOROSHIRO64SSX16 * 3) as *mut _, rng.randiv(v_range, v_min));
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
    v_mult: __m512,
    v_min: __m512,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= XOROSHIRO64SSX16_UNROLL {
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 0), rng.randfv(v_mult, v_min));
            _mm512_stream_ps(out_ptr.add(XOROSHIRO64SSX16 * 1), rng.randfv(v_mult, v_min));
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
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 0), rng.randfv(v_mult, v_min));
            _mm512_storeu_ps(out_ptr.add(XOROSHIRO64SSX16 * 1), rng.randfv(v_mult, v_min));
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
pub extern "C" fn xoroshiro64ssx16_next_u32s(ptr: *mut Xoroshiro64Ssx16, out: *mut u32, count: usize) {
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
                let mut local_rng = Xoroshiro64Ssx16::new(xoroshiro64ssx16_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx16_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn xoroshiro64ssx16_next_f32s(ptr: *mut Xoroshiro64Ssx16, out: *mut f32, count: usize) {
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
                let mut local_rng = Xoroshiro64Ssx16::new(xoroshiro64ssx16_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx16_next_f32s_chunk(&mut local_rng, chunk, scale);
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
        let v_range = _mm512_set1_epi64((max as i64 - min as i64 + 1) as i64);
        let v_min = _mm512_set1_epi32(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOROSHIRO64SSX16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Xoroshiro64Ssx16::new(xoroshiro64ssx16_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx16_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
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
                let mut local_rng = Xoroshiro64Ssx16::new(xoroshiro64ssx16_chunk_seed(base_seed, chunk_idx));
                xoroshiro64ssx16_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}
