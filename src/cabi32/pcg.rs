use crate::dispatch_simd;
use crate::rng::Rng32;
use crate::rng32::{
    PCG32_MULT, PCG32X8_LANE, PCG32X8_PAR_CHUNK, PCG32X8_PAR_CHUNK_BLOCKS, Pcg32, Pcg32Simd,
    Pcg32x8,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::arch::x86_64::*;
use std::slice::from_raw_parts_mut;

/// Creates a new `Pcg32` instance.
/// The caller is responsible for freeing the memory using `pcg32_free`.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_new(seed: u64) -> *mut Pcg32 {
    Box::into_raw(Box::new(Pcg32::new(seed)))
}

/// Frees the memory of a `Pcg32` instance.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_free(ptr: *mut Pcg32) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32_next_u32s(ptr: *mut Pcg32, out: *mut u32, count: usize) {
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
pub extern "C" fn pcg32_next_f32s(ptr: *mut Pcg32, out: *mut f32, count: usize) {
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
pub extern "C" fn pcg32_rand_i32s(
    ptr: *mut Pcg32,
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
pub extern "C" fn pcg32_rand_f32s(
    ptr: *mut Pcg32,
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

#[inline(always)]
fn pcg32_advance_lcg(state: u64, inc: u64, delta: u64) -> u64 {
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    let mut cur_mult = PCG32_MULT;
    let mut cur_plus = inc;
    let mut d = delta;

    while d > 0 {
        if (d & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        d >>= 1;
    }

    state.wrapping_mul(acc_mult).wrapping_add(acc_plus)
}

#[inline(always)]
fn pcg32_advance_coeff(delta: u64) -> (u64, u64) {
    let mut acc_mult = 1u64;
    let mut acc_plus = 0u64;
    let mut cur_mult = PCG32_MULT;
    let mut cur_plus = 1u64;
    let mut d = delta;

    while d > 0 {
        if (d & 1) != 0 {
            acc_mult = acc_mult.wrapping_mul(cur_mult);
            acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
        }
        cur_plus = cur_mult.wrapping_add(1).wrapping_mul(cur_plus);
        cur_mult = cur_mult.wrapping_mul(cur_mult);
        d >>= 1;
    }

    (acc_mult, acc_plus)
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn pcg32x8_next_u32s_chunk(
    chunk: &mut [u32],
    start_state: [u64; PCG32X8_LANE],
    inc0: [u64; PCG32X8_LANE],
    mult_lo: __m512i,
    mult_hi: __m512i,
    mask32: __m512i,
) {
    let mut state = _mm512_loadu_si512(start_state.as_ptr() as *const _);
    let inc = _mm512_loadu_si512(inc0.as_ptr() as *const _);

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks32 = chunk.chunks_exact_mut(PCG32X8_LANE * 4);

    if is_aligned {
        for dst in chunks32.by_ref() {
            let v0 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v1 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v2 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v3 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);

            let res01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
            let res23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3);

            _mm512_stream_si512(dst.as_mut_ptr() as *mut _, res01);
            _mm512_stream_si512(dst[16..].as_mut_ptr() as *mut _, res23);
        }
    } else {
        for dst in chunks32.by_ref() {
            let v0 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v1 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v2 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
            let v3 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);

            let res01 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v0), v1);
            let res23 = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(v2), v3);

            _mm512_storeu_si512(dst.as_mut_ptr() as *mut _, res01);
            _mm512_storeu_si512(dst[16..].as_mut_ptr() as *mut _, res23);
        }
    }

    let rem = chunks32.into_remainder();
    let mut rem_chunks8 = rem.chunks_exact_mut(PCG32X8_LANE);
    for dst in rem_chunks8.by_ref() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, out256);
    }

    let final_rem = rem_chunks8.into_remainder();
    if !final_rem.is_empty() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for j in 0..final_rem.len() {
            final_rem[j] = tmp[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn pcg32x8_next_f32s_chunk(
    chunk: &mut [f32],
    start_state: [u64; PCG32X8_LANE],
    inc0: [u64; PCG32X8_LANE],
    mult_lo: __m512i,
    mult_hi: __m512i,
    mask32: __m512i,
    scale: f32,
) {
    let mut state = _mm512_loadu_si512(start_state.as_ptr() as *const _);
    let inc = _mm512_loadu_si512(inc0.as_ptr() as *const _);

    let mut chunks_exact = chunk.chunks_exact_mut(PCG32X8_LANE);
    for dst in chunks_exact.by_ref() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for i in 0..PCG32X8_LANE {
            dst[i] = tmp[i] as f32 * scale;
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for j in 0..rem.len() {
            rem[j] = tmp[j] as f32 * scale;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn pcg32x8_rand_i32s_chunk(
    chunk: &mut [i32],
    start_state: [u64; PCG32X8_LANE],
    inc0: [u64; PCG32X8_LANE],
    mult_lo: __m512i,
    mult_hi: __m512i,
    mask32: __m512i,
    range: u64,
    min: i32,
) {
    let mut state = _mm512_loadu_si512(start_state.as_ptr() as *const _);
    let inc = _mm512_loadu_si512(inc0.as_ptr() as *const _);

    let mut chunks_exact = chunk.chunks_exact_mut(PCG32X8_LANE);
    for dst in chunks_exact.by_ref() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for i in 0..PCG32X8_LANE {
            dst[i] = ((tmp[i] as u64).wrapping_mul(range) >> 32) as i32 + min;
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for j in 0..rem.len() {
            rem[j] = ((tmp[j] as u64).wrapping_mul(range) >> 32) as i32 + min;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn pcg32x8_rand_f32s_chunk(
    chunk: &mut [f32],
    start_state: [u64; PCG32X8_LANE],
    inc0: [u64; PCG32X8_LANE],
    mult_lo: __m512i,
    mult_hi: __m512i,
    mask32: __m512i,
    scale: f32,
    min: f32,
) {
    let mut state = _mm512_loadu_si512(start_state.as_ptr() as *const _);
    let inc = _mm512_loadu_si512(inc0.as_ptr() as *const _);

    let mut chunks_exact = chunk.chunks_exact_mut(PCG32X8_LANE);
    for dst in chunks_exact.by_ref() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for i in 0..PCG32X8_LANE {
            dst[i] = tmp[i] as f32 * scale + min;
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let out256 = Pcg32x8::step_u32(&mut state, inc, mult_lo, mult_hi, mask32);
        let mut tmp = [0u32; PCG32X8_LANE];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, out256);
        for j in 0..rem.len() {
            rem[j] = tmp[j] as f32 * scale + min;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn pcg32x8_advance_states(
    base_state: [u64; PCG32X8_LANE],
    inc: [u64; PCG32X8_LANE],
    delta: u64,
) -> [u64; PCG32X8_LANE] {
    let mut advanced = [0u64; PCG32X8_LANE];
    for i in 0..PCG32X8_LANE {
        advanced[i] = pcg32_advance_lcg(base_state[i], inc[i], delta);
    }
    advanced
}

#[cfg(target_arch = "x86_64")]
fn pcg32x8_chunk_starts(
    count: usize,
    state0: [u64; PCG32X8_LANE],
    inc0: [u64; PCG32X8_LANE],
) -> Vec<[u64; PCG32X8_LANE]> {
    let num_chunks = count.div_ceil(PCG32X8_PAR_CHUNK);
    let mut starts = Vec::with_capacity(num_chunks);
    if num_chunks == 0 {
        return starts;
    }

    starts.push(state0);
    if num_chunks == 1 {
        return starts;
    }

    let (chunk_mult, chunk_plus_coeff) = pcg32_advance_coeff(PCG32X8_PAR_CHUNK_BLOCKS);
    let mut chunk_plus = [0u64; PCG32X8_LANE];
    for i in 0..PCG32X8_LANE {
        chunk_plus[i] = inc0[i].wrapping_mul(chunk_plus_coeff);
    }

    let mut cur = state0;
    for _ in 1..num_chunks {
        for i in 0..PCG32X8_LANE {
            cur[i] = cur[i].wrapping_mul(chunk_mult).wrapping_add(chunk_plus[i]);
        }
        starts.push(cur);
    }

    starts
}

/// Creates a new `Pcg32x8` instance.
/// The caller is responsible for freeing the memory using `pcg32x8_free`.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_new(seed: u64) -> *mut Pcg32x8 {
    unsafe { Box::into_raw(Box::new(Pcg32x8::new(seed))) }
}

/// Frees the memory of a `Pcg32x8` instance.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_free(ptr: *mut Pcg32x8) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_next_u32s(ptr: *mut Pcg32x8, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut state0 = [0u64; PCG32X8_LANE];
        let mut inc0 = [0u64; PCG32X8_LANE];
        _mm512_storeu_si512(state0.as_mut_ptr() as *mut _, rng.state);
        _mm512_storeu_si512(inc0.as_mut_ptr() as *mut _, rng.inc);

        let mult_lo = _mm512_set1_epi64(0x4C957F2D_i64);
        let mult_hi = _mm512_set1_epi64(0x5851F42D_i64);
        let mask32 = _mm512_set1_epi64(0xFFFFFFFF_i64);
        let chunk_starts = pcg32x8_chunk_starts(count, state0, inc0);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(PCG32X8_PAR_CHUNK)
            .zip(chunk_starts.into_par_iter())
            .for_each(|(chunk, start_state)| {
                pcg32x8_next_u32s_chunk(chunk, start_state, inc0, mult_lo, mult_hi, mask32)
            });

        let num_blocks = ((count + PCG32X8_LANE - 1) / PCG32X8_LANE) as u64;
        state0 = pcg32x8_advance_states(state0, inc0, num_blocks);
        rng.state = _mm512_loadu_si512(state0.as_ptr() as *const _);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_next_f32s(ptr: *mut Pcg32x8, out: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut state0 = [0u64; PCG32X8_LANE];
        let mut inc0 = [0u64; PCG32X8_LANE];
        _mm512_storeu_si512(state0.as_mut_ptr() as *mut _, rng.state);
        _mm512_storeu_si512(inc0.as_mut_ptr() as *mut _, rng.inc);

        let mult_lo = _mm512_set1_epi64(0x4C957F2D_i64);
        let mult_hi = _mm512_set1_epi64(0x5851F42D_i64);
        let mask32 = _mm512_set1_epi64(0xFFFFFFFF_i64);
        let scale = 1.0f32 / (u32::MAX as f32 + 1.0);
        let chunk_starts = pcg32x8_chunk_starts(count, state0, inc0);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PCG32X8_PAR_CHUNK)
            .zip(chunk_starts.into_par_iter())
            .for_each(|(chunk, start_state)| {
                pcg32x8_next_f32s_chunk(chunk, start_state, inc0, mult_lo, mult_hi, mask32, scale)
            });

        let num_blocks = ((count + PCG32X8_LANE - 1) / PCG32X8_LANE) as u64;
        state0 = pcg32x8_advance_states(state0, inc0, num_blocks);
        rng.state = _mm512_loadu_si512(state0.as_ptr() as *const _);
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_rand_i32s(
    ptr: *mut Pcg32x8,
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
        let mut state0 = [0u64; PCG32X8_LANE];
        let mut inc0 = [0u64; PCG32X8_LANE];
        _mm512_storeu_si512(state0.as_mut_ptr() as *mut _, rng.state);
        _mm512_storeu_si512(inc0.as_mut_ptr() as *mut _, rng.inc);

        let mult_lo = _mm512_set1_epi64(0x4C957F2D_i64);
        let mult_hi = _mm512_set1_epi64(0x5851F42D_i64);
        let mask32 = _mm512_set1_epi64(0xFFFFFFFF_i64);
        let range = (max as i64 - min as i64 + 1) as u64;
        let chunk_starts = pcg32x8_chunk_starts(count, state0, inc0);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PCG32X8_PAR_CHUNK)
            .zip(chunk_starts.into_par_iter())
            .for_each(|(chunk, start_state)| {
                pcg32x8_rand_i32s_chunk(
                    chunk,
                    start_state,
                    inc0,
                    mult_lo,
                    mult_hi,
                    mask32,
                    range,
                    min,
                )
            });

        let num_blocks = ((count + PCG32X8_LANE - 1) / PCG32X8_LANE) as u64;
        state0 = pcg32x8_advance_states(state0, inc0, num_blocks);
        rng.state = _mm512_loadu_si512(state0.as_ptr() as *const _);
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn pcg32x8_rand_f32s(
    ptr: *mut Pcg32x8,
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
        let mut state0 = [0u64; PCG32X8_LANE];
        let mut inc0 = [0u64; PCG32X8_LANE];
        _mm512_storeu_si512(state0.as_mut_ptr() as *mut _, rng.state);
        _mm512_storeu_si512(inc0.as_mut_ptr() as *mut _, rng.inc);

        let mult_lo = _mm512_set1_epi64(0x4C957F2D_i64);
        let mult_hi = _mm512_set1_epi64(0x5851F42D_i64);
        let mask32 = _mm512_set1_epi64(0xFFFFFFFF_i64);
        let scale = (max - min) * (1.0f32 / (u32::MAX as f32 + 1.0));
        let chunk_starts = pcg32x8_chunk_starts(count, state0, inc0);

        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(PCG32X8_PAR_CHUNK)
            .zip(chunk_starts.into_par_iter())
            .for_each(|(chunk, start_state)| {
                pcg32x8_rand_f32s_chunk(
                    chunk,
                    start_state,
                    inc0,
                    mult_lo,
                    mult_hi,
                    mask32,
                    scale,
                    min,
                )
            });

        let num_blocks = ((count + PCG32X8_LANE - 1) / PCG32X8_LANE) as u64;
        state0 = pcg32x8_advance_states(state0, inc0, num_blocks);
        rng.state = _mm512_loadu_si512(state0.as_ptr() as *const _);
    }
}

/// Creates a new `Pcg32Simd` instance, dispatching to AVX-512 or scalar implementation.
/// The caller is responsible for freeing the memory using `pcg32simd_free`.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_new(seed: u64) -> *mut Pcg32Simd {
    dispatch_simd!(Pcg32Simd, pcg32_new, pcg32x8_new, seed)
}
/// Frees the memory of a `Pcg32Simd` instance.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_free(ptr: *mut Pcg32Simd) {
    dispatch_simd!(Pcg32x8, Pcg32, pcg32_free, pcg32x8_free, ptr)
}
/// Fills the output buffer with the next random `u32` values using the best available implementation.
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_next_u32s(ptr: *mut Pcg32Simd, out: *mut u32, count: usize) {
    dispatch_simd!(
        Pcg32x8,
        Pcg32,
        pcg32_next_u32s,
        pcg32x8_next_u32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_next_f32s(ptr: *mut Pcg32Simd, out: *mut f32, count: usize) {
    dispatch_simd!(
        Pcg32x8,
        Pcg32,
        pcg32_next_f32s,
        pcg32x8_next_f32s,
        ptr,
        out,
        count
    )
}
/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_rand_i32s(
    ptr: *mut Pcg32Simd,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    dispatch_simd!(
        Pcg32x8,
        Pcg32,
        pcg32_rand_i32s,
        pcg32x8_rand_i32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn pcg32simd_rand_f32s(
    ptr: *mut Pcg32Simd,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    dispatch_simd!(
        Pcg32x8,
        Pcg32,
        pcg32_rand_f32s,
        pcg32x8_rand_f32s,
        ptr,
        out,
        count,
        min,
        max
    )
}
