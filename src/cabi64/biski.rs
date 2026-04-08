use crate::{
    rng::Rng64,
    rng64::{Biski64, SplitMix64, biski::Biski64x8},
};
use rayon::prelude::*;
use std::slice::from_raw_parts_mut;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Creates a new heap-allocated `Biski64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`biski64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn biski64_new(seed: u64) -> *mut Biski64 {
    Box::into_raw(Box::new(Biski64::new(seed)))
}

/// Frees a `Biski64` instance previously created by [`biski64_free`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn biski64_free(ptr: *mut Biski64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const BISKI64_PAR_CHUNK: usize = 4;

#[unsafe(no_mangle)]
pub extern "C" fn biski64_next_u64s(ptr: *mut Biski64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(BISKI64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut rng = Biski64::new(seed);
                for v in chunk {
                    *v = rng.nextu();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64_next_f64s(ptr: *mut Biski64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(BISKI64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut rng = Biski64::new(seed);
                for v in chunk {
                    *v = rng.nextf();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64_rand_i64s(
    ptr: *mut Biski64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(BISKI64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut rng = Biski64::new(seed);
                for v in chunk {
                    *v = rng.randi(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64_rand_f64s(
    ptr: *mut Biski64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(BISKI64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed =
                    SplitMix64::compute(seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE)));
                let mut rng = Biski64::new(seed);
                for v in chunk {
                    *v = rng.randf(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_new(seed: u64) -> *mut Biski64x8 {
    Box::into_raw(Box::new(Biski64x8::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_free(ptr: *mut Biski64x8) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const BISKI64X8_PAR_CHUNK: usize = 0x4000;
const STRIDE: u64 = 0x9E3779B97F4A7C15;

/// 6-way interleaved Biski64x8 with AVX-512 SoA layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn biski64x8_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], seed: u64) {
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));
    let inc = _mm512_set1_epi64(crate::rng64::biski::INC as i64);

    macro_rules! make_state {
        ($group:expr) => {{
            let mut fl = [0u64; 8];
            let mut m = [0u64; 8];
            let mut lm = [0u64; 8];
            for i in 0usize..8 {
                let s = SplitMix64::compute(
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE)),
                );
                let mut sm = SplitMix64::new(s);
                fl[i] = sm.nextu();
                m[i] = sm.nextu();
                lm[i] = sm.nextu();
            }
            (
                _mm512_loadu_si512(fl.as_ptr() as *const __m512i),
                _mm512_loadu_si512(m.as_ptr() as *const __m512i),
                _mm512_loadu_si512(lm.as_ptr() as *const __m512i),
            )
        }};
    }

    let (mut fl0, mut m0, mut lm0) = make_state!(0);
    let (mut fl1, mut m1, mut lm1) = make_state!(1);
    let (mut fl2, mut m2, mut lm2) = make_state!(2);
    let (mut fl3, mut m3, mut lm3) = make_state!(3);
    let (mut fl4, mut m4, mut lm4) = make_state!(4);
    let (mut fl5, mut m5, mut lm5) = make_state!(5);

    macro_rules! step6 {
        () => {{
            // output = mix + loop_mix  (before state update)
            let r0 = _mm512_add_epi64(m0, lm0);
            let r1 = _mm512_add_epi64(m1, lm1);
            let r2 = _mm512_add_epi64(m2, lm2);
            let r3 = _mm512_add_epi64(m3, lm3);
            let r4 = _mm512_add_epi64(m4, lm4);
            let r5 = _mm512_add_epi64(m5, lm5);
            // fast_loop' = fast_loop + INC
            fl0 = _mm512_add_epi64(fl0, inc);
            fl1 = _mm512_add_epi64(fl1, inc);
            fl2 = _mm512_add_epi64(fl2, inc);
            fl3 = _mm512_add_epi64(fl3, inc);
            fl4 = _mm512_add_epi64(fl4, inc);
            fl5 = _mm512_add_epi64(fl5, inc);
            // mix' = rol16(mix) + rol40(loop_mix)
            m0 = _mm512_add_epi64(_mm512_rol_epi64(m0, 16), _mm512_rol_epi64(lm0, 40));
            m1 = _mm512_add_epi64(_mm512_rol_epi64(m1, 16), _mm512_rol_epi64(lm1, 40));
            m2 = _mm512_add_epi64(_mm512_rol_epi64(m2, 16), _mm512_rol_epi64(lm2, 40));
            m3 = _mm512_add_epi64(_mm512_rol_epi64(m3, 16), _mm512_rol_epi64(lm3, 40));
            m4 = _mm512_add_epi64(_mm512_rol_epi64(m4, 16), _mm512_rol_epi64(lm4, 40));
            m5 = _mm512_add_epi64(_mm512_rol_epi64(m5, 16), _mm512_rol_epi64(lm5, 40));
            // loop_mix' = fast_loop' ^ mix'
            lm0 = _mm512_xor_si512(fl0, m0);
            lm1 = _mm512_xor_si512(fl1, m1);
            lm2 = _mm512_xor_si512(fl2, m2);
            lm3 = _mm512_xor_si512(fl3, m3);
            lm4 = _mm512_xor_si512(fl4, m4);
            lm5 = _mm512_xor_si512(fl5, m5);
            (r0, r1, r2, r3, r4, r5)
        }};
    }

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(48);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3, r4, r5) = step6!();
            let p = dst.as_mut_ptr();
            _mm512_stream_si512(p as *mut __m512i, r0);
            _mm512_stream_si512(p.add(8) as *mut __m512i, r1);
            _mm512_stream_si512(p.add(16) as *mut __m512i, r2);
            _mm512_stream_si512(p.add(24) as *mut __m512i, r3);
            _mm512_stream_si512(p.add(32) as *mut __m512i, r4);
            _mm512_stream_si512(p.add(40) as *mut __m512i, r5);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3, r4, r5) = step6!();
            let p = dst.as_mut_ptr();
            _mm512_storeu_si512(p as *mut __m512i, r0);
            _mm512_storeu_si512(p.add(8) as *mut __m512i, r1);
            _mm512_storeu_si512(p.add(16) as *mut __m512i, r2);
            _mm512_storeu_si512(p.add(24) as *mut __m512i, r3);
            _mm512_storeu_si512(p.add(32) as *mut __m512i, r4);
            _mm512_storeu_si512(p.add(40) as *mut __m512i, r5);
        }
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0u64; 8 * 6];
        let (r0, r1, r2, r3, r4, r5) = step6!();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut __m512i, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut __m512i, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut __m512i, r3);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut __m512i, r4);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(40) as *mut __m512i, r5);
        for (i, v) in rem.iter_mut().enumerate() {
            *v = tmp[i];
        }
    }
}

/// Converts 8 u64 (as __m512i) to f64 [0, 1) via bit-manipulation.
/// Uses top-53-bit trick: shift right 11, set exponent to 0x3FF, subtract 1.0.
/// No integer-to-float conversion instruction needed (pure bit ops + sub).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn u64x8_to_f64x8(u: __m512i, exp_bits: __m512i, ones: __m512d) -> __m512d {
    let mantissa = _mm512_srli_epi64(u, 11);
    let bits = _mm512_or_si512(mantissa, exp_bits);
    _mm512_sub_pd(_mm512_castsi512_pd(bits), ones)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn biski64x8_next_f64s_chunk(chunk_idx: usize, chunk: &mut [f64], seed: u64) {
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));
    let inc = _mm512_set1_epi64(crate::rng64::biski::INC as i64);
    let exp_bits = _mm512_set1_epi64(0x3FF0000000000000u64 as i64);
    let ones = _mm512_set1_pd(1.0f64);

    macro_rules! make_state {
        ($group:expr) => {{
            let mut fl = [0u64; 8];
            let mut m = [0u64; 8];
            let mut lm = [0u64; 8];
            for i in 0usize..8 {
                let s = SplitMix64::compute(
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE)),
                );
                let mut sm = SplitMix64::new(s);
                fl[i] = sm.nextu();
                m[i] = sm.nextu();
                lm[i] = sm.nextu();
            }
            (
                _mm512_loadu_si512(fl.as_ptr() as *const __m512i),
                _mm512_loadu_si512(m.as_ptr() as *const __m512i),
                _mm512_loadu_si512(lm.as_ptr() as *const __m512i),
            )
        }};
    }

    let (mut fl0, mut m0, mut lm0) = make_state!(0);
    let (mut fl1, mut m1, mut lm1) = make_state!(1);
    let (mut fl2, mut m2, mut lm2) = make_state!(2);
    let (mut fl3, mut m3, mut lm3) = make_state!(3);
    let (mut fl4, mut m4, mut lm4) = make_state!(4);
    let (mut fl5, mut m5, mut lm5) = make_state!(5);

    macro_rules! step6_f64 {
        () => {{
            let u0 = _mm512_add_epi64(m0, lm0);
            let u1 = _mm512_add_epi64(m1, lm1);
            let u2 = _mm512_add_epi64(m2, lm2);
            let u3 = _mm512_add_epi64(m3, lm3);
            let u4 = _mm512_add_epi64(m4, lm4);
            let u5 = _mm512_add_epi64(m5, lm5);
            fl0 = _mm512_add_epi64(fl0, inc);
            fl1 = _mm512_add_epi64(fl1, inc);
            fl2 = _mm512_add_epi64(fl2, inc);
            fl3 = _mm512_add_epi64(fl3, inc);
            fl4 = _mm512_add_epi64(fl4, inc);
            fl5 = _mm512_add_epi64(fl5, inc);
            m0 = _mm512_add_epi64(_mm512_rol_epi64(m0, 16), _mm512_rol_epi64(lm0, 40));
            m1 = _mm512_add_epi64(_mm512_rol_epi64(m1, 16), _mm512_rol_epi64(lm1, 40));
            m2 = _mm512_add_epi64(_mm512_rol_epi64(m2, 16), _mm512_rol_epi64(lm2, 40));
            m3 = _mm512_add_epi64(_mm512_rol_epi64(m3, 16), _mm512_rol_epi64(lm3, 40));
            m4 = _mm512_add_epi64(_mm512_rol_epi64(m4, 16), _mm512_rol_epi64(lm4, 40));
            m5 = _mm512_add_epi64(_mm512_rol_epi64(m5, 16), _mm512_rol_epi64(lm5, 40));
            lm0 = _mm512_xor_si512(fl0, m0);
            lm1 = _mm512_xor_si512(fl1, m1);
            lm2 = _mm512_xor_si512(fl2, m2);
            lm3 = _mm512_xor_si512(fl3, m3);
            lm4 = _mm512_xor_si512(fl4, m4);
            lm5 = _mm512_xor_si512(fl5, m5);
            (
                u64x8_to_f64x8(u0, exp_bits, ones),
                u64x8_to_f64x8(u1, exp_bits, ones),
                u64x8_to_f64x8(u2, exp_bits, ones),
                u64x8_to_f64x8(u3, exp_bits, ones),
                u64x8_to_f64x8(u4, exp_bits, ones),
                u64x8_to_f64x8(u5, exp_bits, ones),
            )
        }};
    }

    let mut chunks_exact = chunk.chunks_exact_mut(48);
    for dst in chunks_exact.by_ref() {
        let (r0, r1, r2, r3, r4, r5) = step6_f64!();
        let p = dst.as_mut_ptr();
        _mm512_storeu_pd(p, r0);
        _mm512_storeu_pd(p.add(8), r1);
        _mm512_storeu_pd(p.add(16), r2);
        _mm512_storeu_pd(p.add(24), r3);
        _mm512_storeu_pd(p.add(32), r4);
        _mm512_storeu_pd(p.add(40), r5);
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0f64; 48];
        let (r0, r1, r2, r3, r4, r5) = step6_f64!();
        _mm512_storeu_pd(tmp.as_mut_ptr(), r0);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(8), r1);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(16), r2);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(24), r3);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(32), r4);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(40), r5);
        for (j, v) in rem.iter_mut().enumerate() {
            *v = tmp[j];
        }
    }
}

/// 64-bit range mapping: ((u as u128 * range) >> 64) as i64 + min.
/// No SIMD 128-bit multiply available in AVX-512F, so generate u64 in SIMD
/// and apply range in scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn biski64x8_rand_i64s_chunk(
    chunk_idx: usize,
    chunk: &mut [i64],
    seed: u64,
    min: i64,
    max: i64,
) {
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));
    let inc = _mm512_set1_epi64(crate::rng64::biski::INC as i64);
    let range = (max as i128 - min as i128 + 1) as u128;

    macro_rules! make_state {
        ($group:expr) => {{
            let mut fl = [0u64; 8];
            let mut m = [0u64; 8];
            let mut lm = [0u64; 8];
            for i in 0usize..8 {
                let s = SplitMix64::compute(
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE)),
                );
                let mut sm = SplitMix64::new(s);
                fl[i] = sm.nextu();
                m[i] = sm.nextu();
                lm[i] = sm.nextu();
            }
            (
                _mm512_loadu_si512(fl.as_ptr() as *const __m512i),
                _mm512_loadu_si512(m.as_ptr() as *const __m512i),
                _mm512_loadu_si512(lm.as_ptr() as *const __m512i),
            )
        }};
    }

    let (mut fl0, mut m0, mut lm0) = make_state!(0);
    let (mut fl1, mut m1, mut lm1) = make_state!(1);
    let (mut fl2, mut m2, mut lm2) = make_state!(2);
    let (mut fl3, mut m3, mut lm3) = make_state!(3);
    let (mut fl4, mut m4, mut lm4) = make_state!(4);
    let (mut fl5, mut m5, mut lm5) = make_state!(5);

    macro_rules! step6_u64 {
        () => {{
            let r0 = _mm512_add_epi64(m0, lm0);
            let r1 = _mm512_add_epi64(m1, lm1);
            let r2 = _mm512_add_epi64(m2, lm2);
            let r3 = _mm512_add_epi64(m3, lm3);
            let r4 = _mm512_add_epi64(m4, lm4);
            let r5 = _mm512_add_epi64(m5, lm5);
            fl0 = _mm512_add_epi64(fl0, inc);
            fl1 = _mm512_add_epi64(fl1, inc);
            fl2 = _mm512_add_epi64(fl2, inc);
            fl3 = _mm512_add_epi64(fl3, inc);
            fl4 = _mm512_add_epi64(fl4, inc);
            fl5 = _mm512_add_epi64(fl5, inc);
            m0 = _mm512_add_epi64(_mm512_rol_epi64(m0, 16), _mm512_rol_epi64(lm0, 40));
            m1 = _mm512_add_epi64(_mm512_rol_epi64(m1, 16), _mm512_rol_epi64(lm1, 40));
            m2 = _mm512_add_epi64(_mm512_rol_epi64(m2, 16), _mm512_rol_epi64(lm2, 40));
            m3 = _mm512_add_epi64(_mm512_rol_epi64(m3, 16), _mm512_rol_epi64(lm3, 40));
            m4 = _mm512_add_epi64(_mm512_rol_epi64(m4, 16), _mm512_rol_epi64(lm4, 40));
            m5 = _mm512_add_epi64(_mm512_rol_epi64(m5, 16), _mm512_rol_epi64(lm5, 40));
            lm0 = _mm512_xor_si512(fl0, m0);
            lm1 = _mm512_xor_si512(fl1, m1);
            lm2 = _mm512_xor_si512(fl2, m2);
            lm3 = _mm512_xor_si512(fl3, m3);
            lm4 = _mm512_xor_si512(fl4, m4);
            lm5 = _mm512_xor_si512(fl5, m5);
            (r0, r1, r2, r3, r4, r5)
        }};
    }

    #[inline(always)]
    unsafe fn apply_range(tmp: &[u64; 48], dst: &mut [i64], range: u128, min: i64) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = ((tmp[i] as u128 * range) >> 64) as i64 + min;
        }
    }

    let mut chunks_exact = chunk.chunks_exact_mut(48);
    for dst in chunks_exact.by_ref() {
        let (r0, r1, r2, r3, r4, r5) = step6_u64!();
        let mut tmp = [0u64; 48];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut __m512i, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut __m512i, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut __m512i, r3);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut __m512i, r4);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(40) as *mut __m512i, r5);
        apply_range(&tmp, dst, range, min);
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let (r0, r1, r2, r3, r4, r5) = step6_u64!();
        let mut tmp = [0u64; 48];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut __m512i, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut __m512i, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut __m512i, r3);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut __m512i, r4);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(40) as *mut __m512i, r5);
        apply_range(&tmp, rem, range, min);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn biski64x8_rand_f64s_chunk(
    chunk_idx: usize,
    chunk: &mut [f64],
    seed: u64,
    min: f64,
    max: f64,
) {
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));
    let inc = _mm512_set1_epi64(crate::rng64::biski::INC as i64);
    let exp_bits = _mm512_set1_epi64(0x3FF0000000000000u64 as i64);
    let one = _mm512_set1_pd(1.0f64);
    let range = _mm512_set1_pd(max - min);
    let min_pd = _mm512_set1_pd(min);

    macro_rules! make_state {
        ($group:expr) => {{
            let mut fl = [0u64; 8];
            let mut m = [0u64; 8];
            let mut lm = [0u64; 8];
            for i in 0usize..8 {
                let s = SplitMix64::compute(
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE)),
                );
                let mut sm = SplitMix64::new(s);
                fl[i] = sm.nextu();
                m[i] = sm.nextu();
                lm[i] = sm.nextu();
            }
            (
                _mm512_loadu_si512(fl.as_ptr() as *const __m512i),
                _mm512_loadu_si512(m.as_ptr() as *const __m512i),
                _mm512_loadu_si512(lm.as_ptr() as *const __m512i),
            )
        }};
    }

    let (mut fl0, mut m0, mut lm0) = make_state!(0);
    let (mut fl1, mut m1, mut lm1) = make_state!(1);
    let (mut fl2, mut m2, mut lm2) = make_state!(2);
    let (mut fl3, mut m3, mut lm3) = make_state!(3);
    let (mut fl4, mut m4, mut lm4) = make_state!(4);
    let (mut fl5, mut m5, mut lm5) = make_state!(5);

    macro_rules! step6_randf {
        () => {{
            let u0 = _mm512_add_epi64(m0, lm0);
            let u1 = _mm512_add_epi64(m1, lm1);
            let u2 = _mm512_add_epi64(m2, lm2);
            let u3 = _mm512_add_epi64(m3, lm3);
            let u4 = _mm512_add_epi64(m4, lm4);
            let u5 = _mm512_add_epi64(m5, lm5);
            fl0 = _mm512_add_epi64(fl0, inc);
            fl1 = _mm512_add_epi64(fl1, inc);
            fl2 = _mm512_add_epi64(fl2, inc);
            fl3 = _mm512_add_epi64(fl3, inc);
            fl4 = _mm512_add_epi64(fl4, inc);
            fl5 = _mm512_add_epi64(fl5, inc);
            m0 = _mm512_add_epi64(_mm512_rol_epi64(m0, 16), _mm512_rol_epi64(lm0, 40));
            m1 = _mm512_add_epi64(_mm512_rol_epi64(m1, 16), _mm512_rol_epi64(lm1, 40));
            m2 = _mm512_add_epi64(_mm512_rol_epi64(m2, 16), _mm512_rol_epi64(lm2, 40));
            m3 = _mm512_add_epi64(_mm512_rol_epi64(m3, 16), _mm512_rol_epi64(lm3, 40));
            m4 = _mm512_add_epi64(_mm512_rol_epi64(m4, 16), _mm512_rol_epi64(lm4, 40));
            m5 = _mm512_add_epi64(_mm512_rol_epi64(m5, 16), _mm512_rol_epi64(lm5, 40));
            lm0 = _mm512_xor_si512(fl0, m0);
            lm1 = _mm512_xor_si512(fl1, m1);
            lm2 = _mm512_xor_si512(fl2, m2);
            lm3 = _mm512_xor_si512(fl3, m3);
            lm4 = _mm512_xor_si512(fl4, m4);
            lm5 = _mm512_xor_si512(fl5, m5);
            // f = (u >> 11 | exp_bits) as f64 - 1.0, then fma(f, range, min)
            macro_rules! to_randf {
                ($u:expr) => {{
                    let f = u64x8_to_f64x8($u, exp_bits, one);
                    _mm512_fmadd_pd(f, range, min_pd)
                }};
            }
            (
                to_randf!(u0),
                to_randf!(u1),
                to_randf!(u2),
                to_randf!(u3),
                to_randf!(u4),
                to_randf!(u5),
            )
        }};
    }

    let mut chunks_exact = chunk.chunks_exact_mut(48);
    for dst in chunks_exact.by_ref() {
        let (r0, r1, r2, r3, r4, r5) = step6_randf!();
        let p = dst.as_mut_ptr();
        _mm512_storeu_pd(p, r0);
        _mm512_storeu_pd(p.add(8), r1);
        _mm512_storeu_pd(p.add(16), r2);
        _mm512_storeu_pd(p.add(24), r3);
        _mm512_storeu_pd(p.add(32), r4);
        _mm512_storeu_pd(p.add(40), r5);
    }

    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0f64; 48];
        let (r0, r1, r2, r3, r4, r5) = step6_randf!();
        _mm512_storeu_pd(tmp.as_mut_ptr(), r0);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(8), r1);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(16), r2);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(24), r3);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(32), r4);
        _mm512_storeu_pd(tmp.as_mut_ptr().add(40), r5);
        for (j, v) in rem.iter_mut().enumerate() {
            *v = tmp[j];
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_next_u64s(ptr: *mut Biski64x8, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let vals = rng.nextu();
        let seed = vals.iter().copied().fold(0, u64::wrapping_add);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(BISKI64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                biski64x8_next_u64s_chunk(chunk_idx, chunk, seed);
            });

        let seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Biski64x8::new(seed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_next_f64s(ptr: *mut Biski64x8, out: *mut f64, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let vals = rng.nextu();
        let seed = vals.iter().copied().fold(0, u64::wrapping_add);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(BISKI64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                biski64x8_next_f64s_chunk(chunk_idx, chunk, seed);
            });

        let seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Biski64x8::new(seed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_rand_i64s(
    ptr: *mut Biski64x8,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let vals = rng.nextu();
        let seed = vals.iter().copied().fold(0, u64::wrapping_add);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(BISKI64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                biski64x8_rand_i64s_chunk(chunk_idx, chunk, seed, min, max);
            });

        let seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Biski64x8::new(seed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn biski64x8_rand_f64s(
    ptr: *mut Biski64x8,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let vals = rng.nextu();
        let seed = vals.iter().copied().fold(0, u64::wrapping_add);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(BISKI64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                biski64x8_rand_f64s_chunk(chunk_idx, chunk, seed, min, max);
            });

        let seed = SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(STRIDE)));
        *rng = Biski64x8::new(seed);
    }
}
