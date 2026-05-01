use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{ptr, slice::from_raw_parts_mut};
use wide::{f32x4, i32x4, u32x4};

use crate::{
    _internal::FSCALE32,
    rng::Rng32,
    rng32::{
        Sfc32,
        sfc::{SFC32X4, SFC32X8, SFC32X16, Sfc32x4, Sfc32x8, Sfc32x16},
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

// --- Sfc32x4 ---

const SFC32X4_PAR_CHUNK: usize = 0x20000;
const SFC32X4_UNROLL: usize = SFC32X4 << 2;

#[inline(always)]
fn sfc32x4_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
    let x = base_seed.wrapping_add((chunk_idx as u32).wrapping_mul(0x9E37_79B9));
    let mut z = x as u64;
    z ^= z >> 16;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 16;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    (z ^ (z >> 16)) as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_new(seed: u32) -> *mut Sfc32x4 {
    Box::into_raw(Box::new(Sfc32x4::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_free(ptr: *mut Sfc32x4) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn sfc32x4_next_u32s_chunk(rng: &mut Sfc32x4, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    while remaining >= SFC32X4_UNROLL {
        let v0: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        let v1: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        let v2: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        let v3: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        ptr::copy_nonoverlapping(v0.as_ptr(), out_ptr, SFC32X4);
        ptr::copy_nonoverlapping(v1.as_ptr(), out_ptr.add(SFC32X4), SFC32X4);
        ptr::copy_nonoverlapping(v2.as_ptr(), out_ptr.add(SFC32X4 * 2), SFC32X4);
        ptr::copy_nonoverlapping(v3.as_ptr(), out_ptr.add(SFC32X4 * 3), SFC32X4);
        out_ptr = out_ptr.add(SFC32X4_UNROLL);
        remaining -= SFC32X4_UNROLL;
    }
    while remaining >= SFC32X4 {
        let v: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, SFC32X4);
        out_ptr = out_ptr.add(SFC32X4);
        remaining -= SFC32X4;
    }
    if remaining > 0 {
        let v: [u32; SFC32X4] = bytemuck::cast(rng.nextuv());
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn sfc32x4_next_f32s_chunk(rng: &mut Sfc32x4, chunk: &mut [f32], scale: f32x4) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    while remaining >= SFC32X4_UNROLL {
        let v0: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        let v1: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        let v2: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        let v3: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        ptr::copy_nonoverlapping(v0.as_ptr(), out_ptr, SFC32X4);
        ptr::copy_nonoverlapping(v1.as_ptr(), out_ptr.add(SFC32X4), SFC32X4);
        ptr::copy_nonoverlapping(v2.as_ptr(), out_ptr.add(SFC32X4 * 2), SFC32X4);
        ptr::copy_nonoverlapping(v3.as_ptr(), out_ptr.add(SFC32X4 * 3), SFC32X4);
        out_ptr = out_ptr.add(SFC32X4_UNROLL);
        remaining -= SFC32X4_UNROLL;
    }
    while remaining >= SFC32X4 {
        let v: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, SFC32X4);
        out_ptr = out_ptr.add(SFC32X4);
        remaining -= SFC32X4;
    }
    if remaining > 0 {
        let v: [f32; SFC32X4] = bytemuck::cast(rng.nextfv(scale));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn sfc32x4_rand_i32s_chunk(
    rng: &mut Sfc32x4,
    chunk: &mut [i32],
    v_range: u32x4,
    v_min: i32x4,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    while remaining >= SFC32X4_UNROLL {
        let v0: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        let v1: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        let v2: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        let v3: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(v0.as_ptr(), out_ptr, SFC32X4);
        ptr::copy_nonoverlapping(v1.as_ptr(), out_ptr.add(SFC32X4), SFC32X4);
        ptr::copy_nonoverlapping(v2.as_ptr(), out_ptr.add(SFC32X4 * 2), SFC32X4);
        ptr::copy_nonoverlapping(v3.as_ptr(), out_ptr.add(SFC32X4 * 3), SFC32X4);
        out_ptr = out_ptr.add(SFC32X4_UNROLL);
        remaining -= SFC32X4_UNROLL;
    }
    while remaining >= SFC32X4 {
        let v: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, SFC32X4);
        out_ptr = out_ptr.add(SFC32X4);
        remaining -= SFC32X4;
    }
    if remaining > 0 {
        let v: [i32; SFC32X4] = bytemuck::cast(rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn sfc32x4_rand_f32s_chunk(
    rng: &mut Sfc32x4,
    chunk: &mut [f32],
    v_mult: f32x4,
    v_min: f32x4,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    while remaining >= SFC32X4_UNROLL {
        let v0: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        let v1: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        let v2: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        let v3: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(v0.as_ptr(), out_ptr, SFC32X4);
        ptr::copy_nonoverlapping(v1.as_ptr(), out_ptr.add(SFC32X4), SFC32X4);
        ptr::copy_nonoverlapping(v2.as_ptr(), out_ptr.add(SFC32X4 * 2), SFC32X4);
        ptr::copy_nonoverlapping(v3.as_ptr(), out_ptr.add(SFC32X4 * 3), SFC32X4);
        out_ptr = out_ptr.add(SFC32X4_UNROLL);
        remaining -= SFC32X4_UNROLL;
    }
    while remaining >= SFC32X4 {
        let v: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, SFC32X4);
        out_ptr = out_ptr.add(SFC32X4);
        remaining -= SFC32X4;
    }
    if remaining > 0 {
        let v: [f32; SFC32X4] = bytemuck::cast(rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, remaining);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_next_u32s(ptr: *mut Sfc32x4, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = rng.nextu()[0];
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x4::new(sfc32x4_chunk_seed(base_seed, chunk_idx));
                sfc32x4_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_next_f32s(ptr: *mut Sfc32x4, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let base_seed = rng.nextu()[0];
        let scale = f32x4::splat(FSCALE32);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x4::new(sfc32x4_chunk_seed(base_seed, chunk_idx));
                sfc32x4_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_rand_i32s(
    ptr: *mut Sfc32x4,
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
        let base_seed = rng.nextu()[0];
        let v_range = u32x4::splat((max as i64 - min as i64 + 1) as u32);
        let v_min = i32x4::splat(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x4::new(sfc32x4_chunk_seed(base_seed, chunk_idx));
                sfc32x4_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x4_rand_f32s(
    ptr: *mut Sfc32x4,
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
        let base_seed = rng.nextu()[0];
        let v_mult = f32x4::splat((max - min) * FSCALE32);
        let v_min = f32x4::splat(min);
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X4_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x4::new(sfc32x4_chunk_seed(base_seed, chunk_idx));
                sfc32x4_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
    }
}

// --- Sfc32x8 ---

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_new(seed: u32) -> *mut Sfc32x8 {
    unsafe { Box::into_raw(Box::new(Sfc32x8::new(seed))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_free(ptr: *mut Sfc32x8) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
const SFC32X8_PAR_CHUNK: usize = 0x20000;
#[cfg(target_arch = "x86_64")]
const SFC32X8_UNROLL: usize = SFC32X8 << 2;

#[cfg(target_arch = "x86_64")]
#[inline]
fn sfc32x8_chunk_seed(base_seed: u32, chunk_idx: usize) -> u32 {
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
unsafe fn sfc32x8_next_u32s_chunk(rng: &mut Sfc32x8, chunk: &mut [u32]) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X8_UNROLL {
            _mm256_stream_si256(out_ptr.add(SFC32X8 * 0) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(SFC32X8 * 1) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(SFC32X8 * 2) as *mut _, rng.nextuv());
            _mm256_stream_si256(out_ptr.add(SFC32X8 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            let v = rng.nextuv();
            _mm256_stream_si256(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    } else {
        while remaining >= SFC32X8_UNROLL {
            _mm256_storeu_si256(out_ptr.add(SFC32X8 * 0) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(SFC32X8 * 1) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(SFC32X8 * 2) as *mut _, rng.nextuv());
            _mm256_storeu_si256(out_ptr.add(SFC32X8 * 3) as *mut _, rng.nextuv());
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            let v = rng.nextuv();
            _mm256_storeu_si256(out_ptr as *mut _, v);
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    }

    if remaining > 0 {
        let mut tmp = [0u32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x8_next_f32s_chunk(rng: &mut Sfc32x8, chunk: &mut [f32], scale: __m256) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X8_UNROLL {
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 0), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 1), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 2), rng.nextfv(scale));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_stream_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    } else {
        while remaining >= SFC32X8_UNROLL {
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 0), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 1), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 2), rng.nextfv(scale));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 3), rng.nextfv(scale));
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_storeu_ps(out_ptr, rng.nextfv(scale));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; SFC32X8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), rng.nextfv(scale));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x8_rand_i32s_chunk(
    rng: &mut Sfc32x8,
    chunk: &mut [i32],
    v_range: __m256i,
    v_min: __m256i,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X8_UNROLL {
            _mm256_stream_si256(
                out_ptr.add(SFC32X8 * 0) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_stream_si256(
                out_ptr.add(SFC32X8 * 1) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_stream_si256(
                out_ptr.add(SFC32X8 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_stream_si256(
                out_ptr.add(SFC32X8 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_stream_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    } else {
        while remaining >= SFC32X8_UNROLL {
            _mm256_storeu_si256(
                out_ptr.add(SFC32X8 * 0) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_storeu_si256(
                out_ptr.add(SFC32X8 * 1) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_storeu_si256(
                out_ptr.add(SFC32X8 * 2) as *mut _,
                rng.randiv(v_range, v_min),
            );
            _mm256_storeu_si256(
                out_ptr.add(SFC32X8 * 3) as *mut _,
                rng.randiv(v_range, v_min),
            );
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_storeu_si256(out_ptr as *mut _, rng.randiv(v_range, v_min));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    }

    if remaining > 0 {
        let mut tmp = [0i32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.randiv(v_range, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sfc32x8_rand_f32s_chunk(
    rng: &mut Sfc32x8,
    chunk: &mut [f32],
    v_mult: __m256,
    v_min: __m256,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    let aligned = (out_ptr as usize & 63) == 0;

    if aligned {
        while remaining >= SFC32X8_UNROLL {
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 0), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 1), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 2), rng.randfv(v_mult, v_min));
            _mm256_stream_ps(out_ptr.add(SFC32X8 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_stream_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    } else {
        while remaining >= SFC32X8_UNROLL {
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 0), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 1), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 2), rng.randfv(v_mult, v_min));
            _mm256_storeu_ps(out_ptr.add(SFC32X8 * 3), rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(SFC32X8_UNROLL);
            remaining -= SFC32X8_UNROLL;
        }
        while remaining >= SFC32X8 {
            _mm256_storeu_ps(out_ptr, rng.randfv(v_mult, v_min));
            out_ptr = out_ptr.add(SFC32X8);
            remaining -= SFC32X8;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; SFC32X8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), rng.randfv(v_mult, v_min));
        ptr::copy_nonoverlapping(tmp.as_ptr(), out_ptr, remaining);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_next_u32s(ptr: *mut Sfc32x8, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        let base_seed = tmp[0];

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x8::new(sfc32x8_chunk_seed(base_seed, chunk_idx));
                sfc32x8_next_u32s_chunk(&mut local_rng, chunk);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_next_f32s(ptr: *mut Sfc32x8, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut tmp = [0u32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        let base_seed = tmp[0];

        let scale = _mm256_set1_ps(FSCALE32);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x8::new(sfc32x8_chunk_seed(base_seed, chunk_idx));
                sfc32x8_next_f32s_chunk(&mut local_rng, chunk, scale);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_rand_i32s(
    ptr: *mut Sfc32x8,
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
        let mut tmp = [0u32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        let base_seed = tmp[0];

        let v_min = _mm256_set1_epi32(min);
        let v_range = _mm256_set1_epi64x((max as i64 - min as i64 + 1) as i64);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x8::new(sfc32x8_chunk_seed(base_seed, chunk_idx));
                sfc32x8_rand_i32s_chunk(&mut local_rng, chunk, v_range, v_min);
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc32x8_rand_f32s(
    ptr: *mut Sfc32x8,
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
        let mut tmp = [0u32; SFC32X8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, rng.nextuv());
        let base_seed = tmp[0];

        let v_mult = _mm256_set1_ps((max - min) * FSCALE32);
        let v_min = _mm256_set1_ps(min);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC32X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut local_rng = Sfc32x8::new(sfc32x8_chunk_seed(base_seed, chunk_idx));
                sfc32x8_rand_f32s_chunk(&mut local_rng, chunk, v_mult, v_min);
            });
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
            let v0 = rng.nextfv(scale);
            let v1 = rng.nextfv(scale);
            let v2 = rng.nextfv(scale);
            let v3 = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v0);
            _mm512_stream_ps(out_ptr.add(SFC32X16), v1);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_stream_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextfv(scale);
            _mm512_stream_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    } else {
        while remaining >= SFC32X16_UNROLL {
            let v0 = rng.nextfv(scale);
            let v1 = rng.nextfv(scale);
            let v2 = rng.nextfv(scale);
            let v3 = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v0);
            _mm512_storeu_ps(out_ptr.add(SFC32X16), v1);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 2), v2);
            _mm512_storeu_ps(out_ptr.add(SFC32X16 * 3), v3);
            out_ptr = out_ptr.add(SFC32X16_UNROLL);
            remaining -= SFC32X16_UNROLL;
        }
        while remaining >= SFC32X16 {
            let v = rng.nextfv(scale);
            _mm512_storeu_ps(out_ptr, v);
            out_ptr = out_ptr.add(SFC32X16);
            remaining -= SFC32X16;
        }
    }

    if remaining > 0 {
        let mut tmp = [0f32; SFC32X16];
        let v = rng.nextfv(scale);
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
