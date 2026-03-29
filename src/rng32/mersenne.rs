use crate::rng32::SplitMix32;
use crate::{rng::Rng32, rng64::SplitMix64, wrap};
use bytemuck::cast_slice;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::num::Wrapping;
use std::ptr;
use wide::u32x4;

// --- Mt19937 ---

/// A 32-bit Mersenne Twister (MT19937) random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng32::Mt19937;
///
/// let mut rng = Mt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Mt19937 {
    mt: [Wrapping<u32>; MT32_N],
    mti: Wrapping<usize>,
}

const MT32_N: usize = 624;
const MT32_M: usize = 397;
const MT32_MATRIX_A: u32 = 0x9908B0DF;
const MT32_UPPER_MASK: u32 = 0x80000000;
const MT32_LOWER_MASK: u32 = 0x7FFFFFFF;

impl Mt19937 {
    /// Creates a new `Mt19937` instance seeded with the given value.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial seed value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// assert_eq!(rng.nextu(), 460915295);
    /// ```
    pub fn new(seed: u32) -> Self {
        let mut mt = [wrap!(0u32); MT32_N];
        let mut seedgen = SplitMix32::new(seed);
        mt[0] = wrap!(seedgen.nextu());
        for i in 1..MT32_N {
            let prev = mt[i - 1];
            mt[i] = wrap!(1812433253u32) * (prev ^ (prev >> 30)) + wrap!(i as u32);
        }
        Self {
            mt,
            mti: wrap!(MT32_N),
        }
    }

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// assert_eq!(rng.nextu(), 460915295);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        if self.mti.0 >= MT32_N {
            self.twist();
        }
        let mut y = self.mt[self.mti.0];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7).0 & 0x9D2C5680;
        y ^= (y << 15).0 & 0xEFC60000;
        y ^= y >> 18;
        y.0
    }

    #[inline]
    pub fn fill_next_u32s(&mut self, out: &mut [u32]) {
        let mut written = 0;
        while written < out.len() {
            if self.mti.0 >= MT32_N {
                self.twist();
            }

            let idx = self.mti.0;
            let available = MT32_N - idx;
            let take = available.min(out.len() - written);
            let src = &self.mt[idx..idx + take];
            let dst = &mut out[written..written + take];

            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut y = *s;
                y ^= y >> 11;
                y ^= (y << 7).0 & 0x9D2C5680;
                y ^= (y << 15).0 & 0xEFC60000;
                y ^= y >> 18;
                *d = y.0;
            }

            self.mti += wrap!(take);
            written += take;
        }
    }

    fn twist(&mut self) {
        for i in 0..MT32_N - MT32_M {
            let x = (self.mt[i].0 & MT32_UPPER_MASK) | (self.mt[i + 1].0 & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M] ^ wrap!(x_a);
        }
        for i in MT32_N - MT32_M..MT32_N - 1 {
            let x = (self.mt[i].0 & MT32_UPPER_MASK) | (self.mt[i + 1].0 & MT32_LOWER_MASK);
            let mut x_a = x >> 1;
            if x & 1 != 0 {
                x_a ^= MT32_MATRIX_A;
            }
            self.mt[i] = self.mt[i + MT32_M - MT32_N] ^ wrap!(x_a);
        }
        let x = (self.mt[MT32_N - 1].0 & MT32_UPPER_MASK) | (self.mt[0].0 & MT32_LOWER_MASK);
        let mut x_a = x >> 1;
        if x & 1 != 0 {
            x_a ^= MT32_MATRIX_A;
        }
        self.mt[MT32_N - 1] = self.mt[MT32_M - 1] ^ wrap!(x_a);
        self.mti = wrap!(0);
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        ((self.nextu() as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Mt19937;
    ///
    /// let mut rng = Mt19937::new(1);
    /// let val: f32 = rng.randf(0.0, 1.0);
    /// assert!(val >= 0.0 && val < 1.0);
    /// ```
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Mt19937 {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

// --- Sfmt19937 ---

/// A SIMD oriented Fast Mersenne Twister (SFMT) random number generator (32-bit version).
///
/// # Examples
///
/// ```
/// use urng::rng32::Sfmt19937;
///
/// let mut rng = Sfmt19937::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
#[repr(align(16))]
pub struct Sfmt19937 {
    state: [u32x4; SFMT_N],
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

impl Sfmt19937 {
    /// Creates a new `Sfmt19937` instance seeded with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// assert_eq!(rng.nextu(), 2240536539);
    /// ```
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

        let mut state = [u32x4::default(); SFMT_N];
        for i in 0..SFMT_N {
            state[i] = u32x4::from([
                raw_state[4 * i],
                raw_state[4 * i + 1],
                raw_state[4 * i + 2],
                raw_state[4 * i + 3],
            ]);
        }

        let mut rng = Self {
            state,
            idx: SFMT_N * 4, // Force generate on first call. 156 * 4 = 624 u32s
        };
        rng.period_certification();
        rng
    }

    fn gen_rand_all(&mut self) {
        unsafe {
            let ptr = self.state.as_mut_ptr();
            let mut r1 = *ptr.add(SFMT_N - 2);
            let mut r2 = *ptr.add(SFMT_N - 1);

            // Constant mask vector
            let mask = u32x4::from([SFMT_MSK1, SFMT_MSK2, SFMT_MSK3, SFMT_MSK4]);

            for i in 0..(SFMT_N - SFMT_POS1) {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1);

                // x = lshift128(a, SFMT_SL2=1)
                let x: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(a)) << 8);
                // y = rshift128(c=r1, SFMT_SR2=1)
                let y: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(r1)) >> 8);

                let r = a ^ x ^ ((b >> SFMT_SR1) & mask) ^ y ^ (r2 << SFMT_SL1);

                *p_i = r;
                r1 = r2;
                r2 = r;
            }

            for i in (SFMT_N - SFMT_POS1)..SFMT_N {
                let p_i = ptr.add(i);
                let a = *p_i;
                let b = *ptr.add(i + SFMT_POS1 - SFMT_N);

                // x = lshift128(a, SFMT_SL2=1)
                let x: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(a)) << 8);
                // y = rshift128(c=r1, SFMT_SR2=1)
                let y: u32x4 = bytemuck::cast((bytemuck::cast::<_, u128>(r1)) >> 8);

                let r = a ^ x ^ ((b >> SFMT_SR1) & mask) ^ y ^ (r2 << SFMT_SL1);

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

    /// Generates the next random `u32` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// assert_eq!(rng.nextu(), 2240536539);
    /// ```
    #[inline]
    pub fn nextu(&mut self) -> u32 {
        if self.idx >= SFMT_N * 4 {
            self.gen_rand_all();
            self.idx = 0;
        }

        let s: &[u32] = cast_slice(&self.state);
        let val = s[self.idx];
        self.idx += 1;
        val
    }

    #[inline]
    pub fn fill_next_u32s(&mut self, out: &mut [u32]) {
        let mut written = 0;
        while written < out.len() {
            if self.idx >= SFMT_N * 4 {
                self.gen_rand_all();
                self.idx = 0;
            }

            let available = SFMT_N * 4 - self.idx;
            let take = available.min(out.len() - written);

            unsafe {
                ptr::copy_nonoverlapping(
                    (self.state.as_ptr() as *const u32).add(self.idx),
                    out.as_mut_ptr().add(written),
                    take,
                );
            }

            self.idx += take;
            written += take;
        }
    }

    /// Generates the next random `f32` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f32 {
        self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
    }

    /// Generates a random `i32` value in the range [min, max].
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// let val: i32 = rng.randi(0, 10);
    /// assert!(val >= 0 && val <= 10);
    /// ```
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> i32 {
        let range = (max as i64 - min as i64 + 1) as u64;
        let x = self.nextu();
        ((x as u64 * range) >> 32) as i32 + min
    }

    /// Generates a random `f32` value in the range [min, max).
    ///
    /// # Examples
    ///
    /// ```
    /// use urng::rng32::Sfmt19937;
    ///
    /// let mut rng = Sfmt19937::new(1);
    /// let val: f32 = rng.randf(0.0, 1.0);
    /// assert!(val >= 0.0 && val < 1.0);
    /// ```
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> f32 {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        (self.nextu() as f32 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i32 - 1);
        &choices[index as usize]
    }
}

impl Rng32 for Sfmt19937 {
    #[inline]
    fn randi(&mut self, min: i32, max: i32) -> i32 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f32, max: f32) -> f32 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

const DSFMT_LOW_MASK: u64 = 0x000f_ffff_ffff_ffff;
const DSFMT_HIGH_CONST: u64 = 0x3ff0_0000_0000_0000;
const DSFMT_SR: u32 = 12;

#[inline]
const fn idxof(i: usize) -> usize {
    #[cfg(target_endian = "big")]
    {
        i ^ 1
    }
    #[cfg(not(target_endian = "big"))]
    {
        i
    }
}

macro_rules! define_dsfmt_variant {
    (
        $(#[$meta:meta])*
        $name:ident,
        mexp = $mexp:literal,
        n = $n:literal,
        pos1 = $pos1:literal,
        sl1 = $sl1:literal,
        msk1 = $msk1:expr,
        msk2 = $msk2:expr,
        fix1 = $fix1:expr,
        fix2 = $fix2:expr,
        pcv1 = $pcv1:expr,
        pcv2 = $pcv2:expr
    ) => {
        $(#[$meta])*
        #[repr(C)]
        #[repr(align(16))]
        pub struct $name {
            state: [u64; $n * 2 + 2],
            out_buf: [u32; $n * 4],
            idx: usize,
        }

        impl $name {
            pub fn new(seed: u64) -> Self {
                let mut state = [0u64; $n * 2 + 2];

                unsafe {
                    let psfmt32 = std::slice::from_raw_parts_mut(
                        state.as_mut_ptr() as *mut u32,
                        ($n + 1) * 4,
                    );
                    psfmt32[idxof(0)] = seed as u32;
                    for i in 1..(($n + 1) * 4) {
                        let prev = psfmt32[idxof(i - 1)];
                        psfmt32[idxof(i)] = 1812433253u32
                            .wrapping_mul(prev ^ (prev >> 30))
                            .wrapping_add(i as u32);
                    }
                }

                let mut rng = Self {
                    state,
                    out_buf: [0u32; $n * 4],
                    idx: $n * 4,
                };
                rng.initial_mask();
                rng.period_certification();
                rng
            }

            fn initial_mask(&mut self) {
                for x in &mut self.state[..$n * 2] {
                    *x = (*x & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST;
                }
            }

            fn period_certification(&mut self) {
                let tmp0 = self.state[$n * 2] ^ $fix1;
                let tmp1 = self.state[$n * 2 + 1] ^ $fix2;

                let mut inner = (tmp0 & $pcv1) ^ (tmp1 & $pcv2);
                let mut i = 32;
                while i > 0 {
                    inner ^= inner >> i;
                    i >>= 1;
                }

                if (inner & 1) == 1 {
                    return;
                }

                if ($pcv2 & 1) == 1 {
                    self.state[$n * 2 + 1] ^= 1;
                    return;
                }

                let pcv = [$pcv1, $pcv2];
                for lane in (0..=1).rev() {
                    let mut work = 1u64;
                    for _ in 0..64 {
                        if (work & pcv[lane]) != 0 {
                            self.state[$n * 2 + lane] ^= work;
                            return;
                        }
                        work <<= 1;
                    }
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            fn gen_rand_all_scalar(&mut self) {
                unsafe {
                    let p = self.state.as_mut_ptr();
                    let mut lung0 = *p.add($n * 2);
                    let mut lung1 = *p.add($n * 2 + 1);

                    let mut i = 0usize;
                    while i < ($n - $pos1) {
                        let abase = i * 2;
                        let bbase = (i + $pos1) * 2;

                        let a0 = *p.add(abase);
                        let a1 = *p.add(abase + 1);
                        let b0 = *p.add(bbase);
                        let b1 = *p.add(bbase + 1);

                        let prev_l0 = lung0;
                        let prev_l1 = lung1;

                        lung0 = (a0 << $sl1) ^ (prev_l1 >> 32) ^ (prev_l1 << 32) ^ b0;
                        lung1 = (a1 << $sl1) ^ (prev_l0 >> 32) ^ (prev_l0 << 32) ^ b1;

                        *p.add(abase) = (lung0 >> DSFMT_SR) ^ (lung0 & $msk1) ^ a0;
                        *p.add(abase + 1) = (lung1 >> DSFMT_SR) ^ (lung1 & $msk2) ^ a1;
                        i += 1;
                    }

                    while i < $n {
                        let abase = i * 2;
                        let bbase = (i + $pos1 - $n) * 2;

                        let a0 = *p.add(abase);
                        let a1 = *p.add(abase + 1);
                        let b0 = *p.add(bbase);
                        let b1 = *p.add(bbase + 1);

                        let prev_l0 = lung0;
                        let prev_l1 = lung1;

                        lung0 = (a0 << $sl1) ^ (prev_l1 >> 32) ^ (prev_l1 << 32) ^ b0;
                        lung1 = (a1 << $sl1) ^ (prev_l0 >> 32) ^ (prev_l0 << 32) ^ b1;

                        *p.add(abase) = (lung0 >> DSFMT_SR) ^ (lung0 & $msk1) ^ a0;
                        *p.add(abase + 1) = (lung1 >> DSFMT_SR) ^ (lung1 & $msk2) ^ a1;
                        i += 1;
                    }

                    *p.add($n * 2) = lung0;
                    *p.add($n * 2 + 1) = lung1;
                }
            }

            #[cfg(target_arch = "x86_64")]
            #[allow(unsafe_op_in_unsafe_fn)]
            #[target_feature(enable = "sse2")]
            unsafe fn gen_rand_all_sse2(&mut self) {
                let p = self.state.as_mut_ptr() as *mut __m128i;
                let mut lung = *p.add($n);
                let mask = _mm_set_epi64x($msk2 as i64, $msk1 as i64);

                let mut i = 0usize;
                while i < ($n - $pos1) {
                    let a = *p.add(i);
                    let b = *p.add(i + $pos1);

                    let z = _mm_slli_epi64(a, $sl1 as i32);
                    let mut y = _mm_shuffle_epi32(lung, 0x1b);
                    y = _mm_xor_si128(y, _mm_xor_si128(z, b));

                    let mut v = _mm_srli_epi64(y, DSFMT_SR as i32);
                    let w = _mm_and_si128(y, mask);
                    v = _mm_xor_si128(v, a);
                    v = _mm_xor_si128(v, w);

                    *p.add(i) = v;
                    lung = y;
                    i += 1;
                }

                while i < $n {
                    let a = *p.add(i);
                    let b = *p.add(i + $pos1 - $n);

                    let z = _mm_slli_epi64(a, $sl1 as i32);
                    let mut y = _mm_shuffle_epi32(lung, 0x1b);
                    y = _mm_xor_si128(y, _mm_xor_si128(z, b));

                    let mut v = _mm_srli_epi64(y, DSFMT_SR as i32);
                    let w = _mm_and_si128(y, mask);
                    v = _mm_xor_si128(v, a);
                    v = _mm_xor_si128(v, w);

                    *p.add(i) = v;
                    lung = y;
                    i += 1;
                }

                *p.add($n) = lung;
            }

            #[inline]
            fn gen_rand_all(&mut self) {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    self.gen_rand_all_sse2();
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    self.gen_rand_all_scalar();
                }
            }

            #[inline]
            fn write_mixed_block(src: *const u64, dst: *mut u32, count_u64: usize) {
                unsafe {
                    for i in 0..count_u64 {
                        let v = *src.add(i);
                        let lo = v as u32;
                        let hi_mix = ((v >> 32) as u32).rotate_left(11) ^ lo;
                        *dst.add(i * 2) = lo;
                        *dst.add(i * 2 + 1) = hi_mix;
                    }
                }
            }

            #[inline]
            fn rebuild_out_buf(&mut self) {
                Self::write_mixed_block(self.state.as_ptr(), self.out_buf.as_mut_ptr(), $n * 2);
            }

            #[inline]
            pub fn nextu(&mut self) -> u32 {
                if self.idx >= $n * 4 {
                    self.gen_rand_all();
                    self.rebuild_out_buf();
                    self.idx = 0;
                }

                let v = self.out_buf[self.idx];
                self.idx += 1;
                v
            }

            #[inline]
            pub fn fill_next_u32s(&mut self, out: &mut [u32]) {
                let block = $n * 4;
                if out.is_empty() {
                    return;
                }

                let mut dst = out.as_mut_ptr();
                let mut remaining = out.len();

                if self.idx >= block {
                    self.idx = block;
                }

                if self.idx == block {
                    while remaining >= block {
                        self.gen_rand_all();
                        Self::write_mixed_block(self.state.as_ptr(), dst, $n * 2);
                        unsafe {
                            dst = dst.add(block);
                        }
                        remaining -= block;
                    }
                    self.idx = block;
                }

                if remaining == 0 {
                    return;
                }

                if self.idx >= block {
                    self.gen_rand_all();
                    self.rebuild_out_buf();
                    self.idx = 0;
                }

                while remaining > 0 {
                    let available = block - self.idx;
                    let take = available.min(remaining);

                    unsafe {
                        ptr::copy_nonoverlapping(self.out_buf.as_ptr().add(self.idx), dst, take);
                        dst = dst.add(take);
                    }

                    self.idx += take;
                    remaining -= take;

                    if remaining > 0 && self.idx >= block {
                        self.gen_rand_all();
                        self.rebuild_out_buf();
                        self.idx = 0;
                    }
                }
            }

            #[inline]
            pub fn nextf(&mut self) -> f32 {
                self.nextu() as f32 * (1.0 / (u32::MAX as f32 + 1.0))
            }

            #[inline]
            pub fn randi(&mut self, min: i32, max: i32) -> i32 {
                let range = (max as i64 - min as i64 + 1) as u64;
                ((self.nextu() as u64 * range) >> 32) as i32 + min
            }

            #[inline]
            pub fn randf(&mut self, min: f32, max: f32) -> f32 {
                let range = max - min;
                let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
                (self.nextu() as f32 * scale) + min
            }

            #[inline]
            pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
                let index = self.randi(0, choices.len() as i32 - 1);
                &choices[index as usize]
            }
        }

        impl Rng32 for $name {
            #[inline]
            fn randi(&mut self, min: i32, max: i32) -> i32 {
                self.randi(min, max)
            }

            #[inline]
            fn randf(&mut self, min: f32, max: f32) -> f32 {
                self.randf(min, max)
            }

            #[inline]
            fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
                self.choice(choices)
            }
        }
    };
}

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=521.
    Sfmt521,
    mexp = 521,
    n = 4,
    pos1 = 3,
    sl1 = 25,
    msk1 = 0x000f_bfef_ff77_efffu64,
    msk2 = 0x000f_feeb_fbdf_bfdfu64,
    fix1 = 0xcfb3_93d6_6163_8469u64,
    fix2 = 0xc166_8678_83ae_2adbu64,
    pcv1 = 0xccaa_5880_0000_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=1279.
    Sfmt1279,
    mexp = 1279,
    n = 12,
    pos1 = 9,
    sl1 = 19,
    msk1 = 0x000e_fff7_ffdd_ffeeu64,
    msk2 = 0x000f_bfff_fff7_7fffu64,
    fix1 = 0xb666_2762_3d1a_31beu64,
    fix2 = 0x04b6_c511_47b6_109bu64,
    pcv1 = 0x7049_f2da_382a_6aebu64,
    pcv2 = 0xde4c_a84a_4000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=2203.
    Sfmt2203,
    mexp = 2203,
    n = 20,
    pos1 = 7,
    sl1 = 19,
    msk1 = 0x000f_dfff_f5ed_bfffu64,
    msk2 = 0x000f_77ff_ffff_fbfeu64,
    fix1 = 0xb14e_907a_3933_8485u64,
    fix2 = 0xf98f_0735_c637_ef90u64,
    pcv1 = 0x8000_0000_0000_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=4253.
    Sfmt4253,
    mexp = 4253,
    n = 40,
    pos1 = 19,
    sl1 = 19,
    msk1 = 0x0007_b7ff_fef5_feffu64,
    msk2 = 0x000f_fdff_effe_fbfcu64,
    fix1 = 0x8090_1b5f_d7a1_1c65u64,
    fix2 = 0x5a63_ff0e_7cb0_ba74u64,
    pcv1 = 0x1ad2_77be_1200_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=11213.
    Sfmt11213,
    mexp = 11213,
    n = 107,
    pos1 = 37,
    sl1 = 19,
    msk1 = 0x000f_ffff_fdf7_fffdu64,
    msk2 = 0x000d_ffff_fff6_bfffu64,
    fix1 = 0xd0ef_7b7c_75b0_6793u64,
    fix2 = 0x9c50_ff4c_aae0_a641u64,
    pcv1 = 0x8234_c512_07c8_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=44497.
    Sfmt44497,
    mexp = 44497,
    n = 427,
    pos1 = 304,
    sl1 = 19,
    msk1 = 0x000f_f6df_ffff_ffefu64,
    msk2 = 0x0007_ffdd_deef_ff6fu64,
    fix1 = 0x75d9_10f2_35f6_e10eu64,
    fix2 = 0x7b32_158a_edc8_e969u64,
    pcv1 = 0x4c33_56b2_a000_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=86243.
    Sfmt86243,
    mexp = 86243,
    n = 829,
    pos1 = 231,
    sl1 = 13,
    msk1 = 0x000f_fedf_f6ff_ffdfu64,
    msk2 = 0x000f_fff7_fdff_ff7eu64,
    fix1 = 0x1d55_3e77_6b97_5e68u64,
    fix2 = 0x648f_aadf_1416_bf91u64,
    pcv1 = 0x5f2c_d03e_2758_a373u64,
    pcv2 = 0xc0b7_eb84_1000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=132049.
    Sfmt132049,
    mexp = 132049,
    n = 1269,
    pos1 = 371,
    sl1 = 23,
    msk1 = 0x000f_b9f4_eff4_bf77u64,
    msk2 = 0x000f_ffff_bfef_ff37u64,
    fix1 = 0x4ce2_4c0e_4e23_4f3bu64,
    fix2 = 0x6261_2409_b566_5c2du64,
    pcv1 = 0x1812_3288_9145_d000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

define_dsfmt_variant!(
    /// SFMT variant parameterized by dSFMT MEXP=216091.
    Sfmt216091,
    mexp = 216091,
    n = 2077,
    pos1 = 1890,
    sl1 = 23,
    msk1 = 0x000b_f7df_7fef_cfffu64,
    msk2 = 0x000e_7fff_fef7_37ffu64,
    fix1 = 0xd7f9_5a04_764c_27d7u64,
    fix2 = 0x6a48_3861_810b_ebc2u64,
    pcv1 = 0x3af0_a8f3_d560_0000u64,
    pcv2 = 0x0000_0000_0000_0001u64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_works() {
        let mut rng = Mt19937::new(1);
        assert_eq!(rng.nextu(), 460915295);
        assert_eq!(rng.nextf(), 0.068138085);
    }

    #[test]
    fn sfmt19937_works() {
        let mut rng = Sfmt19937::new(1);
        assert_eq!(rng.nextu(), 2240536539);
        assert_eq!(rng.nextf(), 0.89096653);
    }

    macro_rules! smoke_variant {
        ($name:ident) => {{
            let mut rng = $name::new(1);
            let u = rng.nextu();
            let f = rng.nextf();
            assert!(u <= u32::MAX);
            assert!((0.0..1.0).contains(&f));
        }};
    }

    #[test]
    fn dsfmt_param_variants_smoke() {
        smoke_variant!(Sfmt521);
        smoke_variant!(Sfmt1279);
        smoke_variant!(Sfmt2203);
        smoke_variant!(Sfmt4253);
        smoke_variant!(Sfmt11213);
        smoke_variant!(Sfmt44497);
        smoke_variant!(Sfmt86243);
        smoke_variant!(Sfmt132049);
        smoke_variant!(Sfmt216091);
    }
}
