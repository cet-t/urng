use crate::rng::Rng32;
use crate::dispatch_simd;
use crate::rng32::{
    SPLITMIX32_GAMMA, SPLITMIX32x16, SPLITMIX32x16_PAR_CHUNK, SplitMix32, SplitMix32Simd,
    SplitMix32x16,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::arch::x86_64::*;
use std::slice::from_raw_parts_mut;

/// Creates a new `SplitMix32` instance.
/// The caller is responsible for freeing the memory using `splitmix32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_new(seed: u32) -> *mut SplitMix32 {
    Box::into_raw(Box::new(SplitMix32::new(seed)))
}

/// Frees the memory of a `SplitMix32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_free(ptr: *mut SplitMix32) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_next_u32s(ptr: *mut SplitMix32, out: *mut u32, count: usize) {
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
pub extern "C" fn splitmix32_next_f32s(ptr: *mut SplitMix32, out: *mut f32, count: usize) {
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
pub extern "C" fn splitmix32_rand_i32s(
    ptr: *mut SplitMix32,
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

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32_rand_f32s(
    ptr: *mut SplitMix32,
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

/// Creates a new `SplitMix32x16` instance.
/// The caller is responsible for freeing the memory using `splitmix32x16_free`.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32x16_new(seed: u32) -> *mut SplitMix32x16 {
    unsafe { Box::into_raw(Box::new(SplitMix32x16::new(seed))) }
}

/// Frees the memory of a `SplitMix32x16` instance.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32x16_free(ptr: *mut SplitMix32x16) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn splitmix32x16_next_u32s_chunk(chunk_idx: usize, chunk: &mut [u32], state0: __m512i) {
    let offset = ((chunk_idx * SPLITMIX32x16_PAR_CHUNK) as u32).wrapping_mul(SPLITMIX32_GAMMA);
    let mut state = _mm512_add_epi32(state0, _mm512_set1_epi32(offset as i32));
    let step = _mm512_set1_epi32(SPLITMIX32_GAMMA.wrapping_mul(SPLITMIX32x16 as u32) as i32);

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks16 = chunk.chunks_exact_mut(SPLITMIX32x16);

    if is_aligned {
        for dst in chunks16.by_ref() {
            let v = SplitMix32x16::compute(state);
            _mm512_stream_si512(dst.as_mut_ptr() as *mut _, v);
            state = _mm512_add_epi32(state, step);
        }
    } else {
        for dst in chunks16.by_ref() {
            let v = SplitMix32x16::compute(state);
            _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, v);
            state = _mm512_add_epi32(state, step);
        }
    }

    let rem = chunks16.into_remainder();
    if !rem.is_empty() {
        let v = SplitMix32x16::compute(state);
        let mut tmp = [0u32; SPLITMIX32x16];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, v);
        rem.copy_from_slice(&tmp[..rem.len()]);
    }
}

/// Fills the output buffer with the next random `u32` values using AVX-512.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32x16_next_u32s(ptr: *mut SplitMix32x16, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let state0 = rng.state;

        buffer
            .par_chunks_mut(SPLITMIX32x16_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                splitmix32x16_next_u32s_chunk(chunk_idx, chunk, state0);
            });

        rng.state = _mm512_add_epi32(
            rng.state,
            _mm512_set1_epi32((count as u32).wrapping_mul(SPLITMIX32_GAMMA) as i32),
        );
    }
}

/// Creates a new `SplitMix32Simd` instance, dispatching to AVX-512 or scalar implementation.
/// The caller is responsible for freeing the memory using `splitmix32simd_free`.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32simd_new(seed: u32) -> *mut SplitMix32Simd {
    dispatch_simd!(SplitMix32Simd, splitmix32_new, splitmix32x16_new, seed)
}

/// Frees the memory of a `SplitMix32Simd` instance.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32simd_free(ptr: *mut SplitMix32Simd) {
    dispatch_simd!(
        SplitMix32x16,
        SplitMix32,
        splitmix32_free,
        splitmix32x16_free,
        ptr
    )
}

/// Fills the output buffer with the next random `u32` values using the best available implementation.
#[unsafe(no_mangle)]
pub extern "C" fn splitmix32simd_next_u32s(ptr: *mut SplitMix32Simd, out: *mut u32, count: usize) {
    dispatch_simd!(
        SplitMix32x16,
        SplitMix32,
        splitmix32_next_u32s,
        splitmix32x16_next_u32s,
        ptr,
        out,
        count
    )
}
