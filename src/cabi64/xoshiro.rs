use crate::rng::Rng64;
use crate::rng64::{SplitMix64, Xoshiro256Pp, Xoshiro256Ss, Xoshiro256Ssx2};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::arch::x86_64::_mm512_storeu_si512;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Xoshiro256Pp` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256pp_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_new(seed: u64) -> *mut Xoshiro256Pp {
    Box::into_raw(Box::new(Xoshiro256Pp::new(seed)))
}
/// Frees a `Xoshiro256Pp` instance previously created by [`xoshiro256pp_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_free(ptr: *mut Xoshiro256Pp) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256PP_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_u64s(ptr: *mut Xoshiro256Pp, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_next_f64s(ptr: *mut Xoshiro256Pp, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_rand_i64s(
    ptr: *mut Xoshiro256Pp,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256pp_rand_f64s(
    ptr: *mut Xoshiro256Pp,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256PP_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Pp::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
    }
}

/// Creates a new heap-allocated `Xoshiro256Ssx2` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256ssx2_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_new(seed: u64) -> *mut Xoshiro256Ssx2 {
    Box::into_raw(Box::new(Xoshiro256Ssx2::new(seed)))
}
/// Frees a `Xoshiro256Ssx2` instance previously created by [`xoshiro256ssx2_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_free(ptr: *mut Xoshiro256Ssx2) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256SSX2_PAR_CHUNK: usize = 131_072;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn, unused_assignments)]
unsafe fn xoshiro256ssx2_next_u64s_chunk(chunk_idx: usize, chunk: &mut [u64], base_seed: u64) {
    // 4-way interleaved xoshiro256++ with AVX-512 SoA layout.
    // Each group holds 8 independent xoshiro256++ streams (SoA: one __m512i per state word).
    // 4 groups × 8 streams = 32 independent streams total, producing 32 u64 per iteration.
    let stride = 0x9E3779B97F4A7C15u64;
    let chunk_base = base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(stride));

    macro_rules! make_state_vec {
        ($group:expr) => {{
            let mut sv = [[0u64; 8]; 4];
            for i in 0usize..8 {
                let seed = SplitMix64::compute(
                    chunk_base.wrapping_add((($group * 8 + i) as u64).wrapping_mul(stride)),
                );
                let mut sm = SplitMix64::new(seed);
                sv[0][i] = sm.nextu();
                sv[1][i] = sm.nextu();
                sv[2][i] = sm.nextu();
                sv[3][i] = sm.nextu();
            }
            use std::arch::x86_64::_mm512_loadu_si512;
            (
                _mm512_loadu_si512(sv[0].as_ptr() as *const _),
                _mm512_loadu_si512(sv[1].as_ptr() as *const _),
                _mm512_loadu_si512(sv[2].as_ptr() as *const _),
                _mm512_loadu_si512(sv[3].as_ptr() as *const _),
            )
        }};
    }

    let (mut s0_0, mut s1_0, mut s2_0, mut s3_0) = make_state_vec!(0);
    let (mut s0_1, mut s1_1, mut s2_1, mut s3_1) = make_state_vec!(1);
    let (mut s0_2, mut s1_2, mut s2_2, mut s3_2) = make_state_vec!(2);
    let (mut s0_3, mut s1_3, mut s2_3, mut s3_3) = make_state_vec!(3);

    // xoshiro256++ step: res = rotl(s0+s3,23)+s0; update state in place.
    // All 4 groups' sums and rotates are issued together to maximize port-0 utilization.
    macro_rules! step4 {
        () => {{
            use std::arch::x86_64::{
                _mm512_add_epi64, _mm512_rol_epi64, _mm512_slli_epi64, _mm512_xor_si512,
            };
            // Issue 4 sums + 4 rotates together (4 port-0 ROL ops at once)
            let sum0 = _mm512_add_epi64(s0_0, s3_0);
            let sum1 = _mm512_add_epi64(s0_1, s3_1);
            let sum2 = _mm512_add_epi64(s0_2, s3_2);
            let sum3 = _mm512_add_epi64(s0_3, s3_3);
            let rot0 = _mm512_rol_epi64(sum0, 23);
            let rot1 = _mm512_rol_epi64(sum1, 23);
            let rot2 = _mm512_rol_epi64(sum2, 23);
            let rot3 = _mm512_rol_epi64(sum3, 23);
            let res0 = _mm512_add_epi64(rot0, s0_0);
            let res1 = _mm512_add_epi64(rot1, s0_1);
            let res2 = _mm512_add_epi64(rot2, s0_2);
            let res3 = _mm512_add_epi64(rot3, s0_3);
            // Compute t for all groups (slli = port 0,5)
            let t0 = _mm512_slli_epi64(s1_0, 17);
            let t1 = _mm512_slli_epi64(s1_1, 17);
            let t2 = _mm512_slli_epi64(s1_2, 17);
            let t3 = _mm512_slli_epi64(s1_3, 17);
            // State update (all XORs, ports 0,1,5)
            s2_0 = _mm512_xor_si512(s2_0, s0_0);
            s2_1 = _mm512_xor_si512(s2_1, s0_1);
            s2_2 = _mm512_xor_si512(s2_2, s0_2);
            s2_3 = _mm512_xor_si512(s2_3, s0_3);
            s3_0 = _mm512_xor_si512(s3_0, s1_0);
            s3_1 = _mm512_xor_si512(s3_1, s1_1);
            s3_2 = _mm512_xor_si512(s3_2, s1_2);
            s3_3 = _mm512_xor_si512(s3_3, s1_3);
            s1_0 = _mm512_xor_si512(s1_0, s2_0);
            s1_1 = _mm512_xor_si512(s1_1, s2_1);
            s1_2 = _mm512_xor_si512(s1_2, s2_2);
            s1_3 = _mm512_xor_si512(s1_3, s2_3);
            s0_0 = _mm512_xor_si512(s0_0, s3_0);
            s0_1 = _mm512_xor_si512(s0_1, s3_1);
            s0_2 = _mm512_xor_si512(s0_2, s3_2);
            s0_3 = _mm512_xor_si512(s0_3, s3_3);
            s2_0 = _mm512_xor_si512(s2_0, t0);
            s2_1 = _mm512_xor_si512(s2_1, t1);
            s2_2 = _mm512_xor_si512(s2_2, t2);
            s2_3 = _mm512_xor_si512(s2_3, t3);
            // Final 4 ROLs for s3 (4 more port-0 ops)
            s3_0 = _mm512_rol_epi64(s3_0, 45);
            s3_1 = _mm512_rol_epi64(s3_1, 45);
            s3_2 = _mm512_rol_epi64(s3_2, 45);
            s3_3 = _mm512_rol_epi64(s3_3, 45);
            (res0, res1, res2, res3)
        }};
    }

    let is_aligned = (chunk.as_ptr() as usize) & 63 == 0;
    let mut chunks_exact = chunk.chunks_exact_mut(32);

    if is_aligned {
        for dst in chunks_exact.by_ref() {
            use std::arch::x86_64::_mm512_stream_si512;

            let (r0, r1, r2, r3) = step4!();
            let p = dst.as_mut_ptr();
            _mm512_stream_si512(p as *mut _, r0);
            _mm512_stream_si512(p.add(8) as *mut _, r1);
            _mm512_stream_si512(p.add(16) as *mut _, r2);
            _mm512_stream_si512(p.add(24) as *mut _, r3);
        }
    } else {
        for dst in chunks_exact.by_ref() {
            let (r0, r1, r2, r3) = step4!();
            let p = dst.as_mut_ptr();
            _mm512_storeu_si512(p as *mut _, r0);
            _mm512_storeu_si512(p.add(8) as *mut _, r1);
            _mm512_storeu_si512(p.add(16) as *mut _, r2);
            _mm512_storeu_si512(p.add(24) as *mut _, r3);
        }
    }

    // Handle remainder (< 32 elements)
    let rem = chunks_exact.into_remainder();
    if !rem.is_empty() {
        let mut tmp = [0u64; 32];
        let (r0, r1, r2, r3) = step4!();
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, r0);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(8) as *mut _, r1);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(16) as *mut _, r2);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(24) as *mut _, r3);
        for (j, v) in rem.iter_mut().enumerate() {
            *v = tmp[j];
        }
    }
}

/// Fills `out[0..count]` with raw `u64` random values.
/// Uses AVX-512 8-stream SoA with 4-way interleaving and rayon parallelism.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ssx2_next_u64s(ptr: *mut Xoshiro256Ssx2, out: *mut u64, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let mut s_arr = [0u64; 8];
        _mm512_storeu_si512(s_arr.as_mut_ptr() as *mut _, rng.s);
        let base_seed = s_arr[0]
            .wrapping_add(s_arr[1])
            .wrapping_add(s_arr[2])
            .wrapping_add(s_arr[3]);

        let buffer = from_raw_parts_mut(out, count);
        buffer
            .par_chunks_mut(XOSHIRO256SSX2_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                xoshiro256ssx2_next_u64s_chunk(chunk_idx, chunk, base_seed);
            });

        // Advance RNG state so next call produces a different sequence
        let new_seed = SplitMix64::compute(
            base_seed.wrapping_add((count as u64).wrapping_mul(0x9E3779B97F4A7C15)),
        );
        *rng = Xoshiro256Ssx2::new(new_seed);
    }
}

/// Creates a new heap-allocated `Xoshiro256Ss` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`xoshiro256ss_free`].
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_new(seed: u64) -> *mut Xoshiro256Ss {
    Box::into_raw(Box::new(Xoshiro256Ss::new(seed)))
}
/// Frees a `Xoshiro256Ss` instance previously created by [`xoshiro256ss_free`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_free(ptr: *mut Xoshiro256Ss) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

const XOSHIRO256SS_PAR_CHUNK: usize = 4096;

/// Fills `out[0..count]` with raw `u64` random values using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_u64s(ptr: *mut Xoshiro256Ss, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextu();
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_next_f64s(ptr: *mut Xoshiro256Ss, out: *mut f64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.nextf();
                }
            });
    }
}
/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_rand_i64s(
    ptr: *mut Xoshiro256Ss,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randi(min, max);
                }
            });
    }
}
/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel chunk generation.
#[unsafe(no_mangle)]
pub extern "C" fn xoshiro256ss_rand_f64s(
    ptr: *mut Xoshiro256Ss,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let base_seed = rng.nextu();

        buffer
            .par_chunks_mut(XOSHIRO256SS_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let chunk_seed = SplitMix64::compute(
                    base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
                );
                let mut local_rng = Xoshiro256Ss::new(chunk_seed);
                for v in chunk {
                    *v = local_rng.randf(min, max);
                }
            });
    }
}
