use crate::rng::Rng64;
use crate::rng64::{Cet64, Cet64x8, Cet256, Cet256x2, SplitMix64};
use rayon::prelude::*;
use std::slice::from_raw_parts_mut;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const STRIDE: u64 = 0x9E3779B97F4A7C15;
const CET_SP1: u64 = 0xFFFFFFFFFFFFFF43;
const CET_P1: u64 = 0x94D049BB133111EB;

const CET64_PAR_CHUNK: usize = 0x10000;
const CET256_PAR_CHUNK: usize = 0x10000;
const CET64X8_PAR_CHUNK: usize = 0x10000;
const CET256X2_PAR_CHUNK: usize = 0x10000;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cet_mul_sp2_vec(x: __m512i) -> __m512i {
    // CET_SP2 == -229 mod 2^64.
    let x8 = _mm512_slli_epi64(x, 8);
    let x5 = _mm512_slli_epi64(x, 5);
    let x2 = _mm512_slli_epi64(x, 2);
    let t0 = _mm512_sub_epi64(x8, x5);
    let t1 = _mm512_add_epi64(x2, x);
    let t = _mm512_add_epi64(t0, t1);
    _mm512_sub_epi64(_mm512_setzero_si512(), t)
}

/// Creates a new heap-allocated `Cet64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`cet64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn cet64_new(seed: u64) -> *mut Cet64 {
    Box::into_raw(Box::new(Cet64::new(seed)))
}
/// Frees a `Cet64` instance previously created by [`cet64_free`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_free(ptr: *mut Cet64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}
/// Fills `out[0..count]` with raw `u64` random values.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_next_u64s(ptr: *mut Cet64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet64::new(chunk_seed);
                for v in chunk {
                    *v = local.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[0, 1)`.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_next_f64s(ptr: *mut Cet64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet64::new(chunk_seed);
                for v in chunk {
                    *v = local.nextf();
                }
            });
    }
}
/// Fills `out[0..count]` with `i64` values uniformly distributed in `[min, max]`.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_rand_i64s(
    ptr: *mut Cet64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet64::new(chunk_seed);
                for v in chunk {
                    *v = local.randi(min, max);
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values uniformly distributed in `[min, max)`.
#[unsafe(no_mangle)]
pub extern "C" fn cet64_rand_f64s(
    ptr: *mut Cet64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet64::new(chunk_seed);
                for v in chunk {
                    *v = local.randf(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_new(seed: u64) -> *mut Cet256 {
    Box::into_raw(Box::new(Cet256::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_free(ptr: *mut Cet256) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_next_u64s(ptr: *mut Cet256, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET256_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet256::new(chunk_seed);
                for v in chunk {
                    *v = local.nextu();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_next_f64s(ptr: *mut Cet256, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET256_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet256::new(chunk_seed);
                for v in chunk {
                    *v = local.nextf();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_rand_i64s(
    ptr: *mut Cet256,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET256_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet256::new(chunk_seed);
                for v in chunk {
                    *v = local.randi(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256_rand_f64s(
    ptr: *mut Cet256,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu();
        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(CET256_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut local = Cet256::new(chunk_seed);
                for v in chunk {
                    *v = local.randf(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet64x8_new(seed: u64) -> *mut Cet64x8 {
    Box::into_raw(Box::new(Cet64x8::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn cet64x8_free(ptr: *mut Cet64x8) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn cet64x8_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], seed: u64) {
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));
    let sp1 = _mm512_set1_epi64(CET_SP1 as i64);
    let p1 = _mm512_set1_epi64(CET_P1 as i64);

    macro_rules! make_state {
        ($group:expr) => {{
            let mut s = [0u64; 8];
            for i in 0usize..8 {
                let base =
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE));
                s[i] = SplitMix64::compute(base);
            }
            _mm512_loadu_si512(s.as_ptr() as *const __m512i)
        }};
    }

    let mut s0 = make_state!(0);
    let mut s1 = make_state!(1);
    let mut s2 = make_state!(2);
    let mut s3 = make_state!(3);
    let mut s4 = make_state!(4);
    let mut s5 = make_state!(5);
    let mut s6 = make_state!(6);
    let mut s7 = make_state!(7);

    macro_rules! step8 {
        () => {{
            s0 = _mm512_add_epi64(s0, sp1);
            s1 = _mm512_add_epi64(s1, sp1);
            s2 = _mm512_add_epi64(s2, sp1);
            s3 = _mm512_add_epi64(s3, sp1);
            s4 = _mm512_add_epi64(s4, sp1);
            s5 = _mm512_add_epi64(s5, sp1);
            s6 = _mm512_add_epi64(s6, sp1);
            s7 = _mm512_add_epi64(s7, sp1);

            let mut x0 = s0;
            let mut x1 = s1;
            let mut x2 = s2;
            let mut x3 = s3;
            let mut x4 = s4;
            let mut x5 = s5;
            let mut x6 = s6;
            let mut x7 = s7;

            x0 = _mm512_xor_si512(x0, _mm512_srli_epi64(x0, 30));
            x1 = _mm512_xor_si512(x1, _mm512_srli_epi64(x1, 30));
            x2 = _mm512_xor_si512(x2, _mm512_srli_epi64(x2, 30));
            x3 = _mm512_xor_si512(x3, _mm512_srli_epi64(x3, 30));
            x4 = _mm512_xor_si512(x4, _mm512_srli_epi64(x4, 30));
            x5 = _mm512_xor_si512(x5, _mm512_srli_epi64(x5, 30));
            x6 = _mm512_xor_si512(x6, _mm512_srli_epi64(x6, 30));
            x7 = _mm512_xor_si512(x7, _mm512_srli_epi64(x7, 30));

            x0 = cet_mul_sp2_vec(x0);
            x1 = cet_mul_sp2_vec(x1);
            x2 = cet_mul_sp2_vec(x2);
            x3 = cet_mul_sp2_vec(x3);
            x4 = cet_mul_sp2_vec(x4);
            x5 = cet_mul_sp2_vec(x5);
            x6 = cet_mul_sp2_vec(x6);
            x7 = cet_mul_sp2_vec(x7);

            x0 = _mm512_xor_si512(x0, _mm512_srli_epi64(x0, 27));
            x1 = _mm512_xor_si512(x1, _mm512_srli_epi64(x1, 27));
            x2 = _mm512_xor_si512(x2, _mm512_srli_epi64(x2, 27));
            x3 = _mm512_xor_si512(x3, _mm512_srli_epi64(x3, 27));
            x4 = _mm512_xor_si512(x4, _mm512_srli_epi64(x4, 27));
            x5 = _mm512_xor_si512(x5, _mm512_srli_epi64(x5, 27));
            x6 = _mm512_xor_si512(x6, _mm512_srli_epi64(x6, 27));
            x7 = _mm512_xor_si512(x7, _mm512_srli_epi64(x7, 27));

            x0 = _mm512_mullo_epi64(x0, p1);
            x1 = _mm512_mullo_epi64(x1, p1);
            x2 = _mm512_mullo_epi64(x2, p1);
            x3 = _mm512_mullo_epi64(x3, p1);
            x4 = _mm512_mullo_epi64(x4, p1);
            x5 = _mm512_mullo_epi64(x5, p1);
            x6 = _mm512_mullo_epi64(x6, p1);
            x7 = _mm512_mullo_epi64(x7, p1);

            x0 = _mm512_xor_si512(x0, _mm512_srli_epi64(x0, 31));
            x1 = _mm512_xor_si512(x1, _mm512_srli_epi64(x1, 31));
            x2 = _mm512_xor_si512(x2, _mm512_srli_epi64(x2, 31));
            x3 = _mm512_xor_si512(x3, _mm512_srli_epi64(x3, 31));
            x4 = _mm512_xor_si512(x4, _mm512_srli_epi64(x4, 31));
            x5 = _mm512_xor_si512(x5, _mm512_srli_epi64(x5, 31));
            x6 = _mm512_xor_si512(x6, _mm512_srli_epi64(x6, 31));
            x7 = _mm512_xor_si512(x7, _mm512_srli_epi64(x7, 31));

            (x0, x1, x2, x3, x4, x5, x6, x7)
        }};
    }

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(64);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3, r4, r5, r6, r7) = step8!();
            let p = dst.as_mut_ptr();
            _mm512_stream_si512(p as *mut __m512i, r0);
            _mm512_stream_si512(p.add(8) as *mut __m512i, r1);
            _mm512_stream_si512(p.add(16) as *mut __m512i, r2);
            _mm512_stream_si512(p.add(24) as *mut __m512i, r3);
            _mm512_stream_si512(p.add(32) as *mut __m512i, r4);
            _mm512_stream_si512(p.add(40) as *mut __m512i, r5);
            _mm512_stream_si512(p.add(48) as *mut __m512i, r6);
            _mm512_stream_si512(p.add(56) as *mut __m512i, r7);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3, r4, r5, r6, r7) = step8!();
            let p = dst.as_mut_ptr();
            _mm512_storeu_si512(p as *mut __m512i, r0);
            _mm512_storeu_si512(p.add(8) as *mut __m512i, r1);
            _mm512_storeu_si512(p.add(16) as *mut __m512i, r2);
            _mm512_storeu_si512(p.add(24) as *mut __m512i, r3);
            _mm512_storeu_si512(p.add(32) as *mut __m512i, r4);
            _mm512_storeu_si512(p.add(40) as *mut __m512i, r5);
            _mm512_storeu_si512(p.add(48) as *mut __m512i, r6);
            _mm512_storeu_si512(p.add(56) as *mut __m512i, r7);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0u64; 64];
        let (r0, r1, r2, r3, r4, r5, r6, r7) = step8!();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut __m512i, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut __m512i, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut __m512i, r3);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut __m512i, r4);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(40) as *mut __m512i, r5);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(48) as *mut __m512i, r6);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(56) as *mut __m512i, r7);
        for (i, v) in rem.iter_mut().enumerate() {
            *v = tmp[i];
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet64x8_next_u64s(ptr: *mut Cet64x8, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }

    unsafe {
        let rng = &mut *ptr;
        let seed = rng.nextu().iter().copied().fold(0u64, u64::wrapping_add);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(CET64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                cet64x8_next_u64s_chunk(chunk_idx, chunk, seed);
            });

        let next_seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Cet64x8::new(next_seed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256x2_new(seed: u64) -> *mut Cet256x2 {
    Box::into_raw(Box::new(Cet256x2::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256x2_free(ptr: *mut Cet256x2) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cet256x2_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], seed: u64) {
    let chunk_seed =
        SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
    let mut local = Cet256x2::new(chunk_seed);

    let mut chunks_exact = chunk.chunks_exact_mut(2);
    for dst in chunks_exact.by_ref() {
        let v = local.nextu();
        dst[0] = v[0];
        dst[1] = v[1];
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        rem[0] = local.nextu()[0];
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cet256x2_next_u64s(ptr: *mut Cet256x2, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }

    unsafe {
        let rng = &mut *ptr;
        let seed_pair = rng.nextu();
        let seed = seed_pair[0].wrapping_add(seed_pair[1]);
        let buffer = from_raw_parts_mut(out, count);

        buffer
            .par_chunks_mut(CET256X2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                cet256x2_next_u64s_chunk(chunk_idx, chunk, seed);
            });

        let next_seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Cet256x2::new(next_seed);
    }
}
