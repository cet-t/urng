use crate::rng::Rng64;
use crate::rng64::SplitMix64;
use bytemuck;
use std::ptr;

type U32x4 = [u32; 4];

#[inline(always)]
fn xor4(a: U32x4, b: U32x4) -> U32x4 {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[inline(always)]
fn and4(a: U32x4, b: U32x4) -> U32x4 {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}

#[inline(always)]
fn shl4(a: U32x4, shift: u32) -> U32x4 {
    [a[0] << shift, a[1] << shift, a[2] << shift, a[3] << shift]
}

#[inline(always)]
fn shr4(a: U32x4, shift: u32) -> U32x4 {
    [a[0] >> shift, a[1] >> shift, a[2] >> shift, a[3] >> shift]
}

#[inline(always)]
fn lshift128(a: U32x4, bytes: u32) -> U32x4 {
    bytemuck::cast(bytemuck::cast::<_, u128>(a) << (bytes * 8))
}

#[inline(always)]
fn rshift128(a: U32x4, bytes: u32) -> U32x4 {
    bytemuck::cast(bytemuck::cast::<_, u128>(a) >> (bytes * 8))
}

#[inline(always)]
fn sfmt_recursion(
    a: U32x4,
    b: U32x4,
    r1: U32x4,
    r2: U32x4,
    mask: U32x4,
    sl1: u32,
    sl2: u32,
    sr1: u32,
    sr2: u32,
) -> U32x4 {
    xor4(
        xor4(xor4(a, lshift128(a, sl2)), and4(shr4(b, sr1), mask)),
        xor4(rshift128(r1, sr2), shl4(r2, sl1)),
    )
}

// --- Mt1993764 ---

/// A 64-bit Mersenne Twister (MT19937-64) random number generator.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Mt1993764::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Mt1993764 {
    mt: [u64; N],
    mti: usize,
}

const N: usize = 312;
const M: usize = 156;
const MATRIX_A: u64 = 0xB5026F5AA96619E9;
const UPPER_MASK: u64 = 0xFFFFFFFF80000000;
const LOWER_MASK: u64 = 0x7FFFFFFF;

impl Mt1993764 {
    /// Creates a new `Mt1993764` instance seeded via `SplitMix64`.
    ///
    pub fn new(seed: u64) -> Self {
        let mut mt = [0u64; N];
        let mut seedgen = SplitMix64::new(seed);
        mt[0] = seedgen.nextu();
        for i in 1..N {
            let prev = mt[i - 1];
            mt[i] = 6364136223846793005u64
                .wrapping_mul(prev ^ (prev >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, mti: N }
    }

    /// Generates the next random `u64` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= (y >> 29) & 0x5555555555555555;
        y ^= (y << 17) & 0x71D67FFFEDA60000;
        y ^= (y << 37) & 0xFFF7EEE000000000;
        y ^= y >> 43;
        y
    }

    #[inline]
    pub fn fill_next_u64s(&mut self, out: &mut [u64]) {
        let mut written = 0;
        while written < out.len() {
            if self.mti >= N {
                self.twist();
            }

            let idx = self.mti;
            let available = N - idx;
            let take = available.min(out.len() - written);
            let src = &self.mt[idx..idx + take];
            let dst = &mut out[written..written + take];

            for (d, &y) in dst.iter_mut().zip(src.iter()) {
                let mut v = y;
                v ^= (v >> 29) & 0x5555555555555555;
                v ^= (v << 17) & 0x71D67FFFEDA60000;
                v ^= (v << 37) & 0xFFF7EEE000000000;
                v ^= v >> 43;
                *d = v;
            }

            self.mti += take;
            written += take;
        }
    }

    fn twist(&mut self) {
        for i in 0..N - M {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.mt[i] = self.mt[i + M] ^ x_a;
        }
        for i in N - M..N - 1 {
            let x = (self.mt[i] & UPPER_MASK) | (self.mt[i + 1] & LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MATRIX_A;
            }
            self.mt[i] = self.mt[i + M - N] ^ x_a;
        }
        let x = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        let mut x_a = x >> 1;
        if x & 1 != 0 {
            x_a ^= MATRIX_A;
        }
        self.mt[N - 1] = self.mt[M - 1] ^ x_a;
        self.mti = 0;
    }
}

impl Rng64 for Mt1993764 {
    #[inline]
    fn nextu(&mut self) -> u64 {
        self.nextu()
    }
}

// --- Sfmt1993764 ---

/// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Sfmt1993764::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Sfmt1993764 {
    state: [U32x4; SFMT_N],
    idx: usize,
}

const SFMT_N: usize = 156;
const SFMT_POS1: usize = 122;
const SFMT_SL1: u32 = 18;
const SFMT_SR1: u32 = 11;

const SFMT_MSK1: u32 = 0xdfffffef;
const SFMT_MSK2: u32 = 0xddfecb7f;
const SFMT_MSK3: u32 = 0xbffaffff;
const SFMT_MSK4: u32 = 0xbffffff6;
const SFMT_PARITY1: u32 = 0x00000001;
const SFMT_PARITY2: u32 = 0x00000000;
const SFMT_PARITY3: u32 = 0x00000000;
const SFMT_PARITY4: u32 = 0x13c9e684;

impl Sfmt1993764 {
    /// Creates a new `Sfmt1993764` instance seeded via `SplitMix64`.
    ///
    /// The state is period-certified after initialisation.
    ///
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);

        // Initialize state using u32 array for simplicity
        let mut raw_state = [0u32; SFMT_N * 4];
        for i in 0..SFMT_N * 2 {
            // Fill with 64-bit values from SplitMix64
            let s = seedgen.nextu();
            raw_state[2 * i] = s as u32;
            raw_state[2 * i + 1] = (s >> 32) as u32;
        }

        let mut state = [[0u32; 4]; SFMT_N];
        for i in 0..SFMT_N {
            state[i] = [
                raw_state[4 * i],
                raw_state[4 * i + 1],
                raw_state[4 * i + 2],
                raw_state[4 * i + 3],
            ];
        }

        let mut rng = Self {
            state,
            idx: SFMT_N * 2, // Force generate on first call
        };
        rng.period_certification();
        rng
    }

    fn gen_rand_all(&mut self) {
        unsafe {
            let ptr = self.state.as_mut_ptr();
            let mut r1 = *ptr.add(SFMT_N - 2);
            let mut r2 = *ptr.add(SFMT_N - 1);

            let mask = [SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4];

            for i in 0..(SFMT_N - SFMT_POS1) {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1);
                let r = sfmt_recursion(a, b, r1, r2, mask, SFMT_SL1, 1, SFMT_SR1, 1);

                *p_i = r;
                r1 = r2;
                r2 = r;
            }

            for i in (SFMT_N - SFMT_POS1)..SFMT_N {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1 - SFMT_N);
                let r = sfmt_recursion(a, b, r1, r2, mask, SFMT_SL1, 1, SFMT_SR1, 1);

                *p_i = r;
                r1 = r2;
                r2 = r;
            }
        }
    }

    fn period_certification(&mut self) {
        let mut inner = 0;
        let psfmt32 =
            unsafe { std::slice::from_raw_parts(self.state.as_ptr() as *const u32, SFMT_N * 4) };
        let parity = [SFMT_PARITY1, SFMT_PARITY2, SFMT_PARITY3, SFMT_PARITY4];

        for i in 0..4 {
            inner ^= psfmt32[i] & parity[i];
        }
        let mut i = 16;
        while i > 0 {
            inner ^= inner >> i;
            i >>= 1;
        }
        inner &= 1;

        if inner == 1 {
            return;
        }

        let psfmt32_mut = unsafe {
            std::slice::from_raw_parts_mut(self.state.as_mut_ptr() as *mut u32, SFMT_N * 4)
        };

        for i in 0..4 {
            let mut work = 1;
            for _ in 0..32 {
                if (work & parity[i]) != 0 {
                    psfmt32_mut[i] ^= work;
                    return;
                }
                work <<= 1;
            }
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn fill_next_u64s(&mut self, out: &mut [u64]) {
        let mut written = 0;
        while written < out.len() {
            if self.idx >= SFMT_N * 2 {
                self.gen_rand_all();
                self.idx = 0;
            }

            let available = SFMT_N * 2 - self.idx;
            let take = available.min(out.len() - written);

            unsafe {
                ptr::copy_nonoverlapping(
                    (self.state.as_ptr() as *const u64).add(self.idx),
                    out.as_mut_ptr().add(written),
                    take,
                );
            }

            self.idx += take;
            written += take;
        }
    }
}

impl Rng64 for Sfmt1993764 {
    #[inline]
    fn nextu(&mut self) -> u64 {
        if self.idx >= SFMT_N * 2 {
            self.gen_rand_all();
            self.idx = 0;
        }

        let s: &[u64] = bytemuck::cast_slice(&self.state);
        let val = s[self.idx];
        self.idx += 1;
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Mt1993764);
    crate::safe_test!(Sfmt1993764);
}
