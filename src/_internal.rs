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
