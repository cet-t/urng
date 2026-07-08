use std::ptr;

use bytemuck::cast_slice;
use wrapn::{Wrap, wrap};

use crate::{
    rng::{Rng32, Rng64},
    rng32::SplitMix32,
    rng64::SplitMix64,
};

#[allow(non_camel_case_types)]
type u32x4 = [u32; 4];

#[inline(always)]
fn xor4(a: u32x4, b: u32x4) -> u32x4 {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

#[inline(always)]
fn and4(a: u32x4, b: u32x4) -> u32x4 {
    [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
}

#[inline(always)]
fn shl4(a: u32x4, shift: u32) -> u32x4 {
    [a[0] << shift, a[1] << shift, a[2] << shift, a[3] << shift]
}

#[inline(always)]
fn shr4(a: u32x4, shift: u32) -> u32x4 {
    [a[0] >> shift, a[1] >> shift, a[2] >> shift, a[3] >> shift]
}

#[inline(always)]
fn lshift128(a: u32x4, bytes: u32) -> u32x4 {
    bytemuck::cast(bytemuck::cast::<_, u128>(a) << (bytes * 8))
}

#[inline(always)]
fn rshift128(a: u32x4, bytes: u32) -> u32x4 {
    bytemuck::cast(bytemuck::cast::<_, u128>(a) >> (bytes * 8))
}

#[inline(always)]
fn sfmt_recursion(
    a: u32x4,
    b: u32x4,
    r1: u32x4,
    r2: u32x4,
    mask: u32x4,
    sl1: u32,
    sl2: u32,
    sr1: u32,
    sr2: u32,
) -> u32x4 {
    xor4(
        xor4(xor4(a, lshift128(a, sl2)), and4(shr4(b, sr1), mask)),
        xor4(rshift128(r1, sr2), shl4(r2, sl1)),
    )
}

// --- Mt19937 ---

/// A 32-bit Mersenne Twister (MT19937) random number generator.
///
/// # Examples
///
/// ```
/// use urng::*;
///
/// let mut rng = Mt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Mt19937 {
    mt: [Wrap<u32>; MT32_N],
    mti: Wrap<usize>,
}

const MT32_N: usize = 624;
const MT32_M: usize = 397;
const MT32_MATRIX_A: u32 = 0x9908B0DF;
const MT32_UPPER_MASK: u32 = 0x80000000;
const MT32_LOWER_MASK: u32 = 0x7FFFFFFF;

impl Mt19937 {
    /// Creates a new `Mt19937` instance seeded with the given value.
    pub fn new(seed: u32) -> Self {
        let mut mt = [wrap!(0u32); MT32_N];
        let mut seedgen = SplitMix32::new(seed);
        mt[0] = seedgen.nextu().into();
        for i in 1..MT32_N {
            let prev = mt[i - 1];
            mt[i] = (prev ^ (prev >> 30)) * 1812433253u32 + i as u32;
        }
        Self {
            mt,
            mti: MT32_N.into(),
        }
    }

    /// Kept for sequential single-threaded block fills (cabi now fills in parallel).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn fill_next_u32s(&mut self, out: &mut [u32]) {
        let mut written = 0;
        while written < out.len() {
            if self.mti >= MT32_N {
                self.twist();
            }

            let idx = self.mti;
            let available = wrap!(MT32_N) - idx;
            let take = available.min((out.len() - written).into());
            let src = &self.mt[*idx.raw()..*(idx + take).raw()];
            let dst = &mut out[written..written + take.raw()];

            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut y = *s;
                y ^= y >> 11;
                y ^= (y << 7) & 0x9D2C5680;
                y ^= (y << 15) & 0xEFC60000;
                y ^= y >> 18;
                *d = y.0.0;
            }

            self.mti += take;
            written += take.raw();
        }
    }

    fn twist(&mut self) {
        for i in 0..MT32_N - MT32_M {
            let x = (self.mt[i] & MT32_UPPER_MASK) | (self.mt[i + 1] & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M] ^ x_a;
        }
        for i in MT32_N - MT32_M..MT32_N - 1 {
            let x = (self.mt[i] & MT32_UPPER_MASK) | (self.mt[i + 1] & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M - MT32_N] ^ x_a;
        }
        let x = (self.mt[MT32_N - 1] & MT32_UPPER_MASK) | (self.mt[0] & MT32_LOWER_MASK);
        let mut x_a = x >> 1;
        if x & 1 != 0 {
            x_a ^= MT32_MATRIX_A;
        }
        self.mt[MT32_N - 1] = self.mt[MT32_M - 1] ^ x_a;
        self.mti = wrap!(0);
    }
}

impl Rng32 for Mt19937 {
    #[inline]
    fn nextu(&mut self) -> u32 {
        if self.mti >= MT32_N {
            self.twist();
        }
        let mut y = self.mt[*self.mti.raw()];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C5680;
        y ^= (y << 15) & 0xEFC60000;
        y ^= y >> 18;
        *y.raw()
    }
}

// --- Sfmt19937 ---

/// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator (32-bit version).
///
/// # Examples
///
/// ```
/// use urng::*;
///
/// let mut rng = Sfmt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
#[repr(align(16))]
pub struct Sfmt19937 {
    state: [u32x4; SFMT_N],
    idx: Wrap<usize>,
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

impl Sfmt19937 {
    /// Creates a new `Sfmt19937` instance seeded with the given value.
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
            idx: wrap!(SFMT_N * 4), // Force generate on first call. 156 * 4 = 624 u32s
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

        // Verification passed
        if inner == 1 {
            return;
        }

        // Modification for period certification
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

    /// Kept for sequential single-threaded block fills (cabi now fills in parallel).
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn fill_next_u32s(&mut self, out: &mut [u32]) {
        let mut written = 0;
        while written < out.len() {
            if self.idx >= SFMT_N * 4 {
                self.gen_rand_all();
                self.idx = 0.into();
            }

            let available = SFMT_N * 4 - self.idx.raw();
            let take = available.min(out.len() - written);

            unsafe {
                ptr::copy_nonoverlapping(
                    (self.state.as_ptr() as *const u32).add(*self.idx.raw()),
                    out.as_mut_ptr().add(written),
                    take,
                );
            }

            self.idx += take;
            written += take;
        }
    }
}

impl Rng32 for Sfmt19937 {
    #[inline]
    fn nextu(&mut self) -> u32 {
        if self.idx >= SFMT_N * 4 {
            self.gen_rand_all();
            self.idx = 0.into();
        }

        let s: &[u32] = cast_slice(&self.state);
        let val = s[*self.idx.raw()];
        self.idx += 1;
        val
    }
}

macro_rules! define_sfmt_variant {
    (
        mexp = $mexp:literal,
        n = $n:literal,
        pos1 = $pos1:literal,
        sl1 = $sl1:literal,
        sl2 = $sl2:literal,
        sr1 = $sr1:literal,
        sr2 = $sr2:literal,
        msk1 = $msk1:expr,
        msk2 = $msk2:expr,
        msk3 = $msk3:expr,
        msk4 = $msk4:expr,
        parity1 = $parity1:expr,
        parity2 = $parity2:expr,
        parity3 = $parity3:expr,
        parity4 = $parity4:expr $(,)?
    ) => {
        paste::paste! {
            /// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator (32-bit version).
            ///
            /// # Examples
            ///
            #[doc = concat!(
                "```\n",
                "use urng::rng::Rng32;\n",
                "use urng::rng32::",
                stringify!([<Sfmt $mexp>]),
                ";\n\n",
                "let mut rng = ",
                stringify!([<Sfmt $mexp>]),
                "::new(1);\n",
                "let _ = rng.nextu();\n",
                "```"
            )]
            #[repr(C)]
            #[repr(align(16))]
            pub struct [<Sfmt $mexp>] {
                state: [u32x4; $n],
                idx: ::wrapn::Wrap<usize>,
            }

            impl [<Sfmt $mexp>] {
                pub fn new(seed: u64) -> Self {
                    let mut seedgen = SplitMix64::new(seed);
                    let mut raw_state = [0u32; $n * 4];
                    for i in 0..($n * 2) {
                        let s = seedgen.nextu();
                        raw_state[2 * i]     = s as u32;
                        raw_state[2 * i + 1] = (s >> 32) as u32;
                    }
                    let mut state = [[0u32; 4]; $n];
                    for i in 0..$n {
                        state[i] = [
                            raw_state[4 * i],
                            raw_state[4 * i + 1],
                            raw_state[4 * i + 2],
                            raw_state[4 * i + 3],
                        ];
                    }
                    let mut rng = Self { state, idx: ($n * 4).into() };
                    rng.period_certification();
                    rng
                }

                fn gen_rand_all(&mut self) {
                    unsafe {
                        let ptr = self.state.as_mut_ptr();
                        let mut r1 = *ptr.add($n - 2);
                        let mut r2 = *ptr.add($n - 1);
                        let mask = [$msk1, $msk2, $msk3, $msk4];

                        for i in 0..($n - $pos1) {
                            let p_i = ptr.add(i);
                            let a = *p_i;
                            let b = *ptr.add(i + $pos1);
                            let r = sfmt_recursion(
                                a,
                                b,
                                r1,
                                r2,
                                mask,
                                $sl1 as u32,
                                $sl2 as u32,
                                $sr1 as u32,
                                $sr2 as u32,
                            );
                            *p_i = r;
                            r1 = r2;
                            r2 = r;
                        }

                        for i in ($n - $pos1)..$n {
                            let p_i = ptr.add(i);
                            let a = *p_i;
                            let b = *ptr.add(i + $pos1 - $n);
                            let r = sfmt_recursion(
                                a,
                                b,
                                r1,
                                r2,
                                mask,
                                $sl1 as u32,
                                $sl2 as u32,
                                $sr1 as u32,
                                $sr2 as u32,
                            );
                            *p_i = r;
                            r1 = r2;
                            r2 = r;
                        }
                    }
                }

                fn period_certification(&mut self) {
                    let mut inner = 0u32;
                    let psfmt32 = unsafe {
                        std::slice::from_raw_parts(self.state.as_ptr() as *const u32, $n * 4)
                    };
                    let parity = [$parity1, $parity2, $parity3, $parity4];
                    for i in 0..4 {
                        inner ^= psfmt32[i] & parity[i];
                    }
                    let mut shift = 16u32;
                    while shift > 0 {
                        inner ^= inner >> shift;
                        shift >>= 1;
                    }
                    inner &= 1;
                    if inner == 1 {
                        return;
                    }
                    let psfmt32_mut = unsafe {
                        std::slice::from_raw_parts_mut(self.state.as_mut_ptr() as *mut u32, $n * 4)
                    };
                    for i in 0..4 {
                        let mut work = 1u32;
                        for _ in 0..32 {
                            if (work & parity[i]) != 0 {
                                psfmt32_mut[i] ^= work;
                                return;
                            }
                            work <<= 1;
                        }
                    }
                }

                /// Kept for sequential single-threaded block fills (cabi now fills in parallel).
                #[allow(dead_code)]
                #[inline]
                pub(crate) fn fill_next_u32s(&mut self, out: &mut [u32]) {
                    let mut written = 0;
                    while written < out.len() {
                        if self.idx >= $n * 4 {
                            self.gen_rand_all();
                            self.idx = 0.into();
                        }
                        let available = $n * 4 - self.idx.raw();
                        let take = available.min(out.len() - written);
                        unsafe {
                            ptr::copy_nonoverlapping(
                                (self.state.as_ptr() as *const u32).add(*self.idx.raw()),
                                out.as_mut_ptr().add(written),
                                take,
                            );
                        }
                        self.idx += take;
                        written += take;
                    }
                }
            }

            impl Rng32 for [<Sfmt $mexp>] {
                #[inline]
                fn nextu(&mut self) -> u32 {
                    if self.idx >= $n * 4 {
                        self.gen_rand_all();
                        self.idx = 0.into();
                    }
                    let s: &[u32] = cast_slice(&self.state);
                    let val = s[*self.idx.raw()];
                    self.idx += 1;
                    val
                }
            }
        }
    };
}

define_sfmt_variant!(
    mexp = 607,
    n = 5,
    pos1 = 2,
    sl1 = 15,
    sl2 = 3,
    sr1 = 13,
    sr2 = 3,
    msk1 = 0xfdff37ffu32,
    msk2 = 0xef7f3f7du32,
    msk3 = 0xff777b7du32,
    msk4 = 0x7ff7fb2fu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0x00000000u32,
    parity4 = 0x5986f054u32,
);

define_sfmt_variant!(
    mexp = 1279,
    n = 10,
    pos1 = 7,
    sl1 = 14,
    sl2 = 3,
    sr1 = 5,
    sr2 = 1,
    msk1 = 0xf7fefffdu32,
    msk2 = 0x7fefcfffu32,
    msk3 = 0xaff3ef3fu32,
    msk4 = 0xb5ffff7fu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0x00000000u32,
    parity4 = 0x20000000u32,
);

define_sfmt_variant!(
    mexp = 2281,
    n = 18,
    pos1 = 12,
    sl1 = 19,
    sl2 = 1,
    sr1 = 5,
    sr2 = 1,
    msk1 = 0xbff7ffbfu32,
    msk2 = 0xfdfffffeu32,
    msk3 = 0xf7ffef7fu32,
    msk4 = 0xf2f7cbbfu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0x00000000u32,
    parity4 = 0x41dfa600u32,
);

define_sfmt_variant!(
    mexp = 4253,
    n = 34,
    pos1 = 17,
    sl1 = 20,
    sl2 = 1,
    sr1 = 7,
    sr2 = 1,
    msk1 = 0x9f7bffffu32,
    msk2 = 0x9fffff5fu32,
    msk3 = 0x3efffffbu32,
    msk4 = 0xfffff7bbu32,
    parity1 = 0xa8000001u32,
    parity2 = 0xaf5390a3u32,
    parity3 = 0xb740b3f8u32,
    parity4 = 0x6c11486du32,
);

define_sfmt_variant!(
    mexp = 11213,
    n = 88,
    pos1 = 68,
    sl1 = 14,
    sl2 = 3,
    sr1 = 7,
    sr2 = 3,
    msk1 = 0xeffff7fbu32,
    msk2 = 0xffffffefu32,
    msk3 = 0xdfdfbfffu32,
    msk4 = 0x7fffdbfdu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0xe8148000u32,
    parity4 = 0xd0c7afa3u32,
);

define_sfmt_variant!(
    mexp = 44497,
    n = 348,
    pos1 = 330,
    sl1 = 5,
    sl2 = 3,
    sr1 = 9,
    sr2 = 3,
    msk1 = 0xeffffffbu32,
    msk2 = 0xdfbebfffu32,
    msk3 = 0xbfbf7befu32,
    msk4 = 0x9ffd7bffu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0xa3ac4000u32,
    parity4 = 0xecc1327au32,
);

define_sfmt_variant!(
    mexp = 86243,
    n = 674,
    pos1 = 366,
    sl1 = 6,
    sl2 = 7,
    sr1 = 19,
    sr2 = 1,
    msk1 = 0xfdbffbffu32,
    msk2 = 0xbff7ff3fu32,
    msk3 = 0xfd77efffu32,
    msk4 = 0xbf9ff3ffu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0x00000000u32,
    parity4 = 0xe9528d85u32,
);

define_sfmt_variant!(
    mexp = 132049,
    n = 1032,
    pos1 = 110,
    sl1 = 19,
    sl2 = 1,
    sr1 = 21,
    sr2 = 1,
    msk1 = 0xffffbb5fu32,
    msk2 = 0xfb6ebf95u32,
    msk3 = 0xfffefffau32,
    msk4 = 0xcff77fffu32,
    parity1 = 0x00000001u32,
    parity2 = 0x00000000u32,
    parity3 = 0xcb520000u32,
    parity4 = 0xc7e91c7du32,
);

define_sfmt_variant!(
    mexp = 216091,
    n = 1689,
    pos1 = 627,
    sl1 = 11,
    sl2 = 3,
    sr1 = 10,
    sr2 = 1,
    msk1 = 0xbff7bff7u32,
    msk2 = 0xbfffffffu32,
    msk3 = 0xbffffa7fu32,
    msk4 = 0xffddfbfbu32,
    parity1 = 0xf8000001u32,
    parity2 = 0x89e80709u32,
    parity3 = 0x3bd2b64bu32,
    parity4 = 0x0c64b1e4u32,
);

#[cfg(test)]
mod tests {
    use crate::safe_test;

    use super::*;

    safe_test!(Mt19937);
    safe_test!(Sfmt19937);
    safe_test!(Sfmt607);
    safe_test!(Sfmt1279);
    safe_test!(Sfmt2281);
    safe_test!(Sfmt4253);
    safe_test!(Sfmt11213);
    safe_test!(Sfmt44497);
    safe_test!(Sfmt86243);
    safe_test!(Sfmt132049);
    safe_test!(Sfmt216091);
}
