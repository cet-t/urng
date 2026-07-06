use crate::_internal::{fill_chunk_auto, prefer_nt};
use crate::rng32::{Threefry32x2, Threefry32x4};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::slice::from_raw_parts_mut;

/// Creates a new `Threefry32x4` instance.
/// The caller is responsible for freeing the memory using `threefry32x4_free`.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_new(seed: u32) -> *mut Threefry32x4 {
    Box::into_raw(Box::new(Threefry32x4::new(seed)))
}

/// Frees the memory of a `Threefry32x4` instance.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_free(ptr: *mut Threefry32x4) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const THREEFRY32_PAR_CHUNK: usize = 0x20000;

/// Fills `buffer` in parallel from counter-mode Threefry32x4 blocks,
/// mapping each raw `u32` through `map`. Sixteen independent blocks
/// (64 outputs) are batched per generator call: the fixed-trip-count
/// loop lets LLVM auto-vectorize the cipher rounds across blocks, and
/// the non-temporal path streams whole cache lines.
#[inline(always)]
fn fry4_fill<T, M>(buffer: &mut [T], c0: [u32; 4], k: [u32; 5], tw: [u32; 3], map: M)
where
    T: Copy + Default + Send,
    M: Fn(u32) -> T + Sync,
{
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(THREEFRY32_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_base = (chunk_idx * (THREEFRY32_PAR_CHUNK / 4)) as u64;
            let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
            let mut c64 = c0_64.wrapping_add(chunk_base);
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 64];
                    for b in 0..16 {
                        let cc = c64.wrapping_add(b as u64);
                        let c = [cc as u32, (cc >> 32) as u32, c0[2], c0[3]];
                        let r = Threefry32x4::compute(c, &k, &tw);
                        out[b * 4] = map(r[0]);
                        out[b * 4 + 1] = map(r[1]);
                        out[b * 4 + 2] = map(r[2]);
                        out[b * 4 + 3] = map(r[3]);
                    }
                    c64 = c64.wrapping_add(16);
                    out
                });
            }
        });
}

/// Advances the 128-bit counter past everything `fry4_fill` consumed
/// (16 blocks per 64-output batch, so round `count` up to whole batches).
#[inline(always)]
fn fry4_advance(rng: &mut Threefry32x4, count: usize) {
    let num_blocks = (count.div_ceil(64) * 16) as u64;
    let c0_64 = (rng.c[0].value() as u64) | ((rng.c[1].value() as u64) << 32);
    let new_c64 = c0_64.wrapping_add(num_blocks);
    rng.c[0] = (new_c64 as u32).into();
    rng.c[1] = ((new_c64 >> 32) as u32).into();
    if new_c64 < c0_64 {
        let (n_c2, ovf3) = rng.c[2].value().overflowing_add(1);
        rng.c[2] = n_c2.into();
        if ovf3 {
            rng.c[3] += 1;
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_u32s(ptr: *mut Threefry32x4, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        fry4_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()),
            rng.tw.map(|x| x.value()), |x| x);
        fry4_advance(rng, count);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_next_f32s(ptr: *mut Threefry32x4, out: *mut f32, count: usize) {
    const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        fry4_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()),
            rng.tw.map(|x| x.value()), |x| x as f32 * SCALE);
        fry4_advance(rng, count);
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_i32s(
    ptr: *mut Threefry32x4,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let range = (max as i64 - min as i64 + 1) as u64;
        fry4_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()),
            rng.tw.map(|x| x.value()), |x| {
            ((x as u64 * range) >> 32) as i32 + min
        });
        fry4_advance(rng, count);
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x4_rand_f32s(
    ptr: *mut Threefry32x4,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let mult = (max - min) * SCALE;
        fry4_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()),
            rng.tw.map(|x| x.value()), |x| x as f32 * mult + min);
        fry4_advance(rng, count);
    }
}

/// Creates a new `Threefry32x2` instance.
/// The caller is responsible for freeing the memory using `threefry32x2_free`.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_new(seed: u32) -> *mut Threefry32x2 {
    Box::into_raw(Box::new(Threefry32x2::new(seed)))
}

/// Frees the memory of a `Threefry32x2` instance.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_free(ptr: *mut Threefry32x2) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

const THREEFRY32X2_PAR_CHUNK: usize = 0x20000;

/// Fills `buffer` in parallel from counter-mode Threefry32x2 blocks,
/// mapping each raw `u32` through `map`. Thirty-two independent blocks
/// (64 outputs) are batched per generator call: the fixed-trip-count
/// loop lets LLVM auto-vectorize the cipher rounds across blocks, and
/// the non-temporal path streams whole cache lines.
#[inline(always)]
fn fry2_fill<T, M>(buffer: &mut [T], c0: [u32; 2], k: [u32; 3], map: M)
where
    T: Copy + Default + Send,
    M: Fn(u32) -> T + Sync,
{
    let nt = prefer_nt::<T>(buffer.len());
    buffer
        .par_chunks_mut(THREEFRY32X2_PAR_CHUNK)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let chunk_base = (chunk_idx * (THREEFRY32X2_PAR_CHUNK / 2)) as u64;
            let c0_64 = (c0[0] as u64) | ((c0[1] as u64) << 32);
            let mut c64 = c0_64.wrapping_add(chunk_base);
            unsafe {
                fill_chunk_auto(chunk, nt, || {
                    let mut out = [T::default(); 64];
                    for b in 0..32 {
                        let cc = c64.wrapping_add(b as u64);
                        let c = [cc as u32, (cc >> 32) as u32];
                        let r = Threefry32x2::compute(c, &k);
                        out[b * 2] = map(r[0]);
                        out[b * 2 + 1] = map(r[1]);
                    }
                    c64 = c64.wrapping_add(32);
                    out
                });
            }
        });
}

/// Advances the 64-bit counter past everything `fry2_fill` consumed
/// (32 blocks per 64-output batch).
#[inline(always)]
fn fry2_advance(rng: &mut Threefry32x2, count: usize) {
    let num_blocks = (count.div_ceil(64) * 32) as u64;
    let c0_64 = (rng.c[0].value() as u64) | ((rng.c[1].value() as u64) << 32);
    let new_c64 = c0_64.wrapping_add(num_blocks);
    rng.c[0] = (new_c64 as u32).into();
    rng.c[1] = ((new_c64 >> 32) as u32).into();
}

/// Fills the output buffer with the next random `u32` values.
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_u32s(ptr: *mut Threefry32x2, out: *mut u32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        fry2_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()), |x| x);
        fry2_advance(rng, count);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_next_f32s(ptr: *mut Threefry32x2, out: *mut f32, count: usize) {
    const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        fry2_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()), |x| x as f32 * SCALE);
        fry2_advance(rng, count);
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
/// This function uses parallel processing for large counts.
#[unsafe(no_mangle)]
pub extern "C" fn threefry32x2_rand_i32s(
    ptr: *mut Threefry32x2,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        let range = (max as i64 - min as i64 + 1) as u64;
        fry2_fill(
            buffer,
            rng.c.map(|x| x.value()),
            rng.k.map(|x| x.value()), |x| {
            ((x as u64 * range) >> 32) as i32 + min
        });
        fry2_advance(rng, count);
    }
}
