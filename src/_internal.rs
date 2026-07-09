#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const FSCALE64: f64 = 1.0 / (u64::MAX as f64 + 1.0);
pub const FSCALE32: f32 = 1.0 / (u32::MAX as f32 + 1.0);

/// Implements [`crate::rng::Rng32`] for a counter-based block generator that owns
/// `buf: [Wrap<u32>; N]` and `pos: Wrap<usize>` fields, by buffering blocks
/// produced by an existing `fn $raw(&mut self) -> [u32; N]` method and handing
/// out one scalar per call (recomputing a fresh block every `N`th call).
///
/// Crate-internal only — call as `crate::_internal::impl_ring_rng32!`.
macro_rules! impl_ring_rng32 {
    ($ty:ty, $n:expr, $raw:ident) => {
        impl $crate::rng::Rng32 for $ty {
            #[inline]
            fn nextu(&mut self) -> u32 {
                if self.pos >= $n {
                    self.buf = self.$raw().map(::core::convert::Into::into);
                    self.pos = 0.into();
                }
                let v = self.buf[self.pos.value()];
                self.pos += 1;
                v.value()
            }
        }
    };
}
pub(crate) use impl_ring_rng32;

/// Implements [`crate::rng::Rng64`] for a counter-based block generator; see [`impl_ring_rng32`].
///
/// Crate-internal only — call as `crate::_internal::impl_ring_rng64!`.
macro_rules! impl_ring_rng64 {
    ($ty:ty, $n:expr, $raw:ident) => {
        impl $crate::rng::Rng64 for $ty {
            #[inline]
            fn nextu(&mut self) -> u64 {
                if self.pos >= $n {
                    self.buf = self.$raw().map(::core::convert::Into::into);
                    self.pos = 0.into();
                }
                let v = self.buf[self.pos.value()];
                self.pos += 1;
                v.value()
            }
        }
    };
}
pub(crate) use impl_ring_rng64;

static DEFAULT_SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Derives a time-based 64-bit seed for `Default` impls.
///
/// Mixes wall-clock nanoseconds with a call counter (to avoid identical
/// seeds for back-to-back calls within the same timer tick) through the
/// existing [`crate::rng64::SplitMix64::compute`] finalizer.
pub(crate) fn default_seed64() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let count = DEFAULT_SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    crate::rng64::SplitMix64::compute(nanos ^ count.wrapping_mul(0x9E3779B97F4A7C15))
}

/// 32-bit counterpart of [`default_seed64`]: folds the 64-bit mix down via xor.
pub(crate) fn default_seed32() -> u32 {
    let z = default_seed64();
    (z ^ (z >> 32)) as u32
}

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
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn fill_chunk_nt<T: Copy, const N: usize, F: FnMut() -> [T; N]>(
    chunk: &mut [T],
    mut generate: F,
) {
    use std::arch::x86_64::*;
    let words = (N * size_of::<T>()) / 32;
    // A batch smaller than one 32-byte store cannot be streamed; silently
    // dropping stores would corrupt the output, so fall back instead.
    if words == 0 || !(N * size_of::<T>()).is_multiple_of(32) {
        return fill_chunk(chunk, generate);
    }
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

/// Buffers larger than this are streamed past the cache (NT stores);
/// smaller ones use cached stores so an L3-resident working set never
/// touches DRAM. Compare against the *whole* destination buffer size,
/// not the per-thread chunk.
pub(crate) const NT_THRESHOLD_BYTES: usize = 24 << 20;

/// Returns whether a fill of `total_bytes` should use non-temporal stores.
#[inline(always)]
pub(crate) fn prefer_nt<T>(total_elems: usize) -> bool {
    total_elems * size_of::<T>() > NT_THRESHOLD_BYTES
}

/// [`prefer_nt`] with the element type inferred from a sample slice
/// (typically the per-thread chunk; pass the *total* element count).
#[inline(always)]
pub(crate) fn prefer_nt_for<T>(total_elems: usize, _sample: &[T]) -> bool {
    total_elems * size_of::<T>() > NT_THRESHOLD_BYTES
}

/// Dispatches to [`fill_chunk_nt`] when requested and AVX2 is available,
/// else [`fill_chunk`]. Pass `nt = prefer_nt::<T>(buffer.len())` computed
/// on the whole destination buffer.
///
/// # Safety
///
/// Same contract as [`fill_chunk`].
#[inline(always)]
pub(crate) unsafe fn fill_chunk_auto<T: Copy, const N: usize, F: FnMut() -> [T; N]>(
    chunk: &mut [T],
    nt: bool,
    generate: F,
) {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if nt && std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { fill_chunk_nt(chunk, generate) };
    }
    let _ = nt;
    unsafe { fill_chunk(chunk, generate) }
}

/// Parallel bulk fill for sequential 32-bit-seeded RNGs: each 512KB chunk
/// runs its own RNG reseeded via [`chunk_seed32`], 16 outputs are batched
/// per generator call (64B for 4-byte `T`), and stores are size-adaptive
/// (cached for L3-resident buffers, non-temporal for larger ones).
#[cfg(feature = "cabi")]
pub(crate) fn par_fill_reseed32<R, T, NF, SF>(
    buffer: &mut [T],
    base_seed: u32,
    new_rng: NF,
    step: SF,
) where
    T: Copy + Default + Send,
    NF: Fn(u32) -> R + Sync,
    SF: Fn(&mut R) -> T + Sync,
{
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;
    const PAR_CHUNK: usize = 0x20000;
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut rng = new_rng(chunk_seed32(base_seed, chunk_idx));
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 16];
                    for v in &mut out {
                        *v = step(&mut rng);
                    }
                    out
                });
            }
        });
}

/// 64-bit counterpart of [`par_fill_reseed32`]: 8 outputs per batch
/// (64B for 8-byte `T`), chunk seeds decorrelated by golden-ratio steps
/// plus the SplitMix64 finalizer.
#[cfg(feature = "cabi")]
pub(crate) fn par_fill_reseed64<R, T, NF, SF>(
    buffer: &mut [T],
    base_seed: u64,
    new_rng: NF,
    step: SF,
) where
    T: Copy + Default + Send,
    NF: Fn(u64) -> R + Sync,
    SF: Fn(&mut R) -> T + Sync,
{
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};
    use rayon::slice::ParallelSliceMut;
    const PAR_CHUNK: usize = 0x20000;
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_seed = crate::rng64::SplitMix64::compute(
                base_seed.wrapping_add((chunk_idx as u64).wrapping_mul(0x9E3779B97F4A7C15)),
            );
            let mut rng = new_rng(chunk_seed);
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 8];
                    for v in &mut out {
                        *v = step(&mut rng);
                    }
                    out
                });
            }
        });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_seed_calls_are_distinct() {
        assert_ne!(default_seed64(), default_seed64());
        assert_ne!(default_seed32(), default_seed32());
    }

    /// Sequential counter generator: makes dropped or duplicated batches
    /// detectable as exact-value mismatches.
    fn counter_batches<const N: usize>() -> impl FnMut() -> [u32; N] {
        let mut next = 0u32;
        move || {
            let mut out = [0u32; N];
            for v in &mut out {
                *v = next;
                next += 1;
            }
            out
        }
    }

    fn check_fill(buf: &[u32], len: usize) {
        for (i, &v) in buf[..len].iter().enumerate() {
            assert_eq!(v as usize, i, "element {i} wrong");
        }
    }

    #[test]
    fn fill_chunk_writes_every_element() {
        for len in [0usize, 1, 7, 16, 63, 64, 65, 1000] {
            let mut buf = vec![u32::MAX; len];
            unsafe { fill_chunk::<u32, 16, _>(&mut buf, counter_batches()) };
            check_fill(&buf, len);
        }
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn fill_chunk_nt_writes_every_element() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for len in [0usize, 1, 7, 16, 63, 64, 65, 1000] {
            let mut buf = vec![u32::MAX; len];
            unsafe { fill_chunk_nt::<u32, 16, _>(&mut buf, counter_batches()) };
            check_fill(&buf, len);
        }
    }

    /// Batches smaller than one 32-byte store must fall back to cached
    /// stores instead of silently dropping every store (regression test:
    /// sfc32x4 once produced 4-element batches and wrote nothing).
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn fill_chunk_nt_small_batch_falls_back() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut buf = vec![u32::MAX; 100];
        unsafe { fill_chunk_nt::<u32, 4, _>(&mut buf, counter_batches()) };
        check_fill(&buf, 100);
    }
}
