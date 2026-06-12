pub const FSCALE64: f64 = 1.0 / (u64::MAX as f64 + 1.0);
pub const FSCALE32: f32 = 1.0 / (u32::MAX as f32 + 1.0);

/// Fills `out[..count]` from repeated calls to `next`.
///
/// # Safety
///
/// `out` must be valid for writes of `count` elements.
#[inline(always)]
pub(crate) unsafe fn fill_with<T, F: FnMut() -> T>(out: *mut T, count: usize, mut next: F) {
    let buffer = unsafe { std::slice::from_raw_parts_mut(out, count) };
    for v in buffer {
        *v = next();
    }
}

/// Fills `chunk` from repeated calls to `gen`, 4x unrolled.
/// The tail is filled from one extra `gen` call truncated to the remainder.
#[inline(always)]
pub(crate) unsafe fn fill_chunk<T: Copy, const N: usize, F: FnMut() -> [T; N]>(
    chunk: &mut [T],
    mut generate: F,
) {
    let mut out_ptr = chunk.as_mut_ptr();
    let mut remaining = chunk.len();
    while remaining >= N * 4 {
        let v0 = generate();
        let v1 = generate();
        let v2 = generate();
        let v3 = generate();
        unsafe {
            std::ptr::copy_nonoverlapping(v0.as_ptr(), out_ptr, N);
            std::ptr::copy_nonoverlapping(v1.as_ptr(), out_ptr.add(N), N);
            std::ptr::copy_nonoverlapping(v2.as_ptr(), out_ptr.add(N * 2), N);
            std::ptr::copy_nonoverlapping(v3.as_ptr(), out_ptr.add(N * 3), N);
            out_ptr = out_ptr.add(N * 4);
        }
        remaining -= N * 4;
    }
    while remaining >= N {
        let v = generate();
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, N);
            out_ptr = out_ptr.add(N);
        }
        remaining -= N;
    }
    if remaining > 0 {
        let v = generate();
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr(), out_ptr, remaining) };
    }
}

/// Non-temporal variant of [`fill_chunk`]: streams each generated batch
/// straight to memory, bypassing the cache (no RFO read traffic).
/// Requires `N * size_of::<T>()` to be a multiple of 32 bytes.
/// Falls back to cached stores when `chunk` is not 32-byte aligned.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn fill_chunk_nt<T: Copy, const N: usize, F: FnMut() -> [T; N]>(
    chunk: &mut [T],
    mut generate: F,
) {
    use std::arch::x86_64::*;
    debug_assert!((N * size_of::<T>()) % 32 == 0);
    let words = (N * size_of::<T>()) / 32;
    let mut p = chunk.as_mut_ptr();
    let mut rem = chunk.len();
    if (p as usize) & 31 == 0 {
        while rem >= N {
            let v = generate();
            let src = v.as_ptr() as *const __m256i;
            for i in 0..words {
                _mm256_stream_si256(
                    (p as *mut u8).add(i * 32) as *mut __m256i,
                    _mm256_loadu_si256(src.add(i)),
                );
            }
            p = p.add(N);
            rem -= N;
        }
        _mm_sfence();
    } else {
        while rem >= N {
            let v = generate();
            std::ptr::copy_nonoverlapping(v.as_ptr(), p, N);
            p = p.add(N);
            rem -= N;
        }
    }
    if rem > 0 {
        let v = generate();
        std::ptr::copy_nonoverlapping(v.as_ptr(), p, rem);
    }
}

/// Dispatches to [`fill_chunk_nt`] when AVX2 is available, else [`fill_chunk`].
///
/// # Safety
///
/// Same contract as [`fill_chunk`].
#[inline(always)]
pub(crate) unsafe fn fill_chunk_auto<T: Copy, const N: usize, F: FnMut() -> [T; N]>(
    chunk: &mut [T],
    generate: F,
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { fill_chunk_nt(chunk, generate) };
    }
    unsafe { fill_chunk(chunk, generate) }
}

/// Derives a decorrelated per-chunk seed for parallel buffer fills
/// (golden-ratio sequence + 64-bit avalanche mix).
#[inline]
pub(crate) fn chunk_seed32(base_seed: u32, chunk_idx: usize) -> u32 {
    let x = base_seed.wrapping_add((chunk_idx as u32).wrapping_mul(0x9E37_79B9));
    let mut z = x as u64;
    z ^= z >> 16;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 16;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    (z ^ (z >> 16)) as u32
}
