use crate::rng64::{Sfc64, Sfc64x8, SplitMix64};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::slice::from_raw_parts_mut;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Creates a new heap-allocated `Sfc64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`sfc64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_new(seed: u64) -> *mut Sfc64 {
    Box::into_raw(Box::new(Sfc64::new(seed)))
}

/// Frees a `Sfc64` instance previously created by [`sfc64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn sfc64_free(ptr: *mut Sfc64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const SFC64_PAR_CHUNK: usize = 0x10000;

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_u64s(ptr: *mut Sfc64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed = SplitMix64::compute(
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut rng = Sfc64::new(seed);
                for v in chunk {
                    *v = rng.nextu();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_next_f64s(ptr: *mut Sfc64, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let seed = rng.nextu();

        buffer
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed = SplitMix64::compute(
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut rng = Sfc64::new(seed);
                for v in chunk {
                    *v = rng.nextf();
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_rand_i64s(
    ptr: *mut Sfc64,
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
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed = SplitMix64::compute(
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut rng = Sfc64::new(seed);
                for v in chunk {
                    *v = rng.randi(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64_rand_f64s(
    ptr: *mut Sfc64,
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
            .par_chunks_mut(SFC64_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let seed = SplitMix64::compute(
                    seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut rng = Sfc64::new(seed);
                for v in chunk {
                    *v = rng.randf(min, max);
                }
            });
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64x8_new(seed: u64) -> *mut Sfc64x8 {
    Box::into_raw(Box::new(Sfc64x8::new(seed)))
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64x8_free(ptr: *mut Sfc64x8) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const SFC64X8_PAR_CHUNK: usize = 0x1000;

/// 6-way interleaved SFC64 with AVX-512 SoA layout.
///
/// SFC64's critical path is 5 cycles (res→c→b→res), so 6 independent groups
/// fully hide the latency. Each group holds 8 streams → 48 u64 per iteration.
///
/// Register budget: 6×4=24 state ZMM + 6 res + 1 `one` = 31 ZMM (fits in 32).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn sfc64x8_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], seed: u64) {
    const STRIDE: u64 = 0x9E3779B97F4A7C15u64;
    let chunk_base = seed.wrapping_add((chunk_idx as u64).wrapping_mul(STRIDE));

    macro_rules! make_state {
        ($group:expr) => {{
            let mut va = [0; 8];
            let mut vb = [0; 8];
            let mut vc = [0; 8];
            for i in 0usize..8 {
                let seed = SplitMix64::compute(
                    chunk_base.wrapping_add(((($group << 3) + i) as u64).wrapping_mul(STRIDE)),
                );
                let mut sm = SplitMix64::new(seed);
                va[i] = sm.nextu();
                vb[i] = sm.nextu();
                vc[i] = sm.nextu();
            }
            (
                _mm512_loadu_si512(va.as_ptr() as *const __m512i),
                _mm512_loadu_si512(vb.as_ptr() as *const __m512i),
                _mm512_loadu_si512(vc.as_ptr() as *const __m512i),
                _mm512_set1_epi64(1),
            )
        }};
    }

    let (mut a0, mut b0, mut c0, mut ctr0) = make_state!(0);
    let (mut a1, mut b1, mut c1, mut ctr1) = make_state!(1);
    let (mut a2, mut b2, mut c2, mut ctr2) = make_state!(2);
    let (mut a3, mut b3, mut c3, mut ctr3) = make_state!(3);
    let (mut a4, mut b4, mut c4, mut ctr4) = make_state!(4);
    let (mut a5, mut b5, mut c5, mut ctr5) = make_state!(5);

    let one = _mm512_set1_epi64(1i64);

    macro_rules! step6 {
        () => {{
            let res0 = _mm512_add_epi64(_mm512_add_epi64(a0, b0), ctr0);
            let res1 = _mm512_add_epi64(_mm512_add_epi64(a1, b1), ctr1);
            let res2 = _mm512_add_epi64(_mm512_add_epi64(a2, b2), ctr2);
            let res3 = _mm512_add_epi64(_mm512_add_epi64(a3, b3), ctr3);
            let res4 = _mm512_add_epi64(_mm512_add_epi64(a4, b4), ctr4);
            let res5 = _mm512_add_epi64(_mm512_add_epi64(a5, b5), ctr5);
            // a = b ^ (b >> 11)
            a0 = _mm512_xor_si512(b0, _mm512_srli_epi64(b0, 11));
            a1 = _mm512_xor_si512(b1, _mm512_srli_epi64(b1, 11));
            a2 = _mm512_xor_si512(b2, _mm512_srli_epi64(b2, 11));
            a3 = _mm512_xor_si512(b3, _mm512_srli_epi64(b3, 11));
            a4 = _mm512_xor_si512(b4, _mm512_srli_epi64(b4, 11));
            a5 = _mm512_xor_si512(b5, _mm512_srli_epi64(b5, 11));
            // b = c + (c << 3)
            b0 = _mm512_add_epi64(c0, _mm512_slli_epi64(c0, 3));
            b1 = _mm512_add_epi64(c1, _mm512_slli_epi64(c1, 3));
            b2 = _mm512_add_epi64(c2, _mm512_slli_epi64(c2, 3));
            b3 = _mm512_add_epi64(c3, _mm512_slli_epi64(c3, 3));
            b4 = _mm512_add_epi64(c4, _mm512_slli_epi64(c4, 3));
            b5 = _mm512_add_epi64(c5, _mm512_slli_epi64(c5, 3));
            // c = rotl(res, 24)
            c0 = _mm512_rol_epi64(res0, 24);
            c1 = _mm512_rol_epi64(res1, 24);
            c2 = _mm512_rol_epi64(res2, 24);
            c3 = _mm512_rol_epi64(res3, 24);
            c4 = _mm512_rol_epi64(res4, 24);
            c5 = _mm512_rol_epi64(res5, 24);
            // counter += 1
            ctr0 = _mm512_add_epi64(ctr0, one);
            ctr1 = _mm512_add_epi64(ctr1, one);
            ctr2 = _mm512_add_epi64(ctr2, one);
            ctr3 = _mm512_add_epi64(ctr3, one);
            ctr4 = _mm512_add_epi64(ctr4, one);
            ctr5 = _mm512_add_epi64(ctr5, one);
            (res0, res1, res2, res3, res4, res5)
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
        let mut tmp = [0u64; 48];
        let (r0, r1, r2, r3, r4, r5) = step6!();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut __m512i, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut __m512i, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut __m512i, r3);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut __m512i, r4);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(40) as *mut __m512i, r5);
        for (j, v) in rem.iter_mut().enumerate() {
            *v = tmp[j];
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn sfc64x8_next_u64s(ptr: *mut Sfc64x8, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let vals = rng.nextu();
        let seed = vals.iter().copied().fold(0, u64::wrapping_add);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(SFC64X8_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                sfc64x8_next_u64s_chunk(chunk_idx, chunk, seed);
            });

        let seed =
            SplitMix64::compute(seed.wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15)));
        *rng = Sfc64x8::new(seed);
    }
}
