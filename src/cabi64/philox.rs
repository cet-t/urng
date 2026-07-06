use crate::_internal::{fill_chunk_auto, prefer_nt};
use crate::rng64::Philox64;
use rayon::prelude::*;
use std::slice::from_raw_parts_mut;

/// Creates a new heap-allocated `Philox64` and returns a raw pointer to it.
/// The caller is responsible for freeing it with [`philox64_free`].
#[unsafe(no_mangle)]
pub extern "C" fn philox64_new(seed: u64) -> *mut Philox64 {
    Box::into_raw(Box::new(Philox64::new(seed)))
}
/// Frees a `Philox64` instance previously created by [`philox64_new`].
/// Does nothing if `ptr` is null.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_free(ptr: *mut Philox64) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) }
    }
}

/// Parallel batch size for Philox64 (elements per thread task).
const PHILOX64_PAR_CHUNK: usize = 0x20000;

/// Fills `buffer` in parallel from counter-mode Philox64 blocks, mapping
/// each raw `u64` through `map`. Four blocks (8 outputs = one 64-byte
/// cache line for 8-byte `T`) are batched per generator call so the
/// non-temporal path can stream whole lines.
#[inline(always)]
fn philox64_fill<T, M>(buffer: &mut [T], c0: [u64; 2], k: [u64; 2], map: M)
where
    T: Copy + Default + Send,
    M: Fn(u64) -> T + Sync,
{
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(PHILOX64_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let mut block = ((chunk_idx * PHILOX64_PAR_CHUNK) / 2) as u64;
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 8];
                    for blk in out.chunks_exact_mut(2) {
                        let mut c = c0;
                        let (new_c0, overflow) = c[0].overflowing_add(block);
                        c[0] = new_c0;
                        if overflow {
                            c[1] = c[1].wrapping_add(1);
                        }
                        let r = Philox64::compute(c, k);
                        blk[0] = map(r[0]);
                        blk[1] = map(r[1]);
                        block = block.wrapping_add(1);
                    }
                    out
                });
            }
        });
}

/// Advances the 128-bit counter past everything `philox64_fill` consumed
/// (4 blocks per 8-output batch).
#[inline(always)]
fn philox64_advance(rng: &mut Philox64, count: usize) {
    let num_blocks = (count.div_ceil(8) * 4) as u64;
    let (new_c0, carry) = rng.c[0].value().overflowing_add(num_blocks);
    rng.c[0] = new_c0.into();
    if carry {
        rng.c[1] += 1;
    }
}

/// Fills `out[0..count]` with raw `u64` random values using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_u64s(ptr: *mut Philox64, out: *mut u64, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        philox64_fill(buffer, rng.c.map(|x| x.value()), rng.k.map(|x| x.value()), |x| x);
        philox64_advance(rng, count);
    }
}

/// Fills `out[0..count]` with `f64` values in `[0, 1)` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_next_f64s(ptr: *mut Philox64, out: *mut f64, count: usize) {
    const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        philox64_fill(buffer, rng.c.map(|x| x.value()), rng.k.map(|x| x.value()), |x| x as f64 * SCALE);
        philox64_advance(rng, count);
    }
}

/// Fills `out[0..count]` with `i64` values in `[min, max]` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_i64s(
    ptr: *mut Philox64,
    out: *mut i64,
    count: usize,
    min: i64,
    max: i64,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let range = (max as i128 - min as i128 + 1) as u128;
        philox64_fill(buffer, rng.c.map(|x| x.value()), rng.k.map(|x| x.value()), |x| {
            ((x as u128 * range) >> 64) as i64 + min
        });
        philox64_advance(rng, count);
    }
}

/// Fills `out[0..count]` with `f64` values in `[min, max)` using parallel counter-based generation.
#[unsafe(no_mangle)]
pub extern "C" fn philox64_rand_f64s(
    ptr: *mut Philox64,
    out: *mut f64,
    count: usize,
    min: f64,
    max: f64,
) {
    const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mult = (max - min) * SCALE;
        philox64_fill(buffer, rng.c.map(|x| x.value()), rng.k.map(|x| x.value()), |x| x as f64 * mult + min);
        philox64_advance(rng, count);
    }
}
