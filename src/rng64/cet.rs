use crate::rng::Rng64;
use crate::rng64::SplitMix64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A 64-bit Self-made random number generator.
///
/// This generator uses a 4-cell cellular automaton state and a Weyl counter.
/// It is designed for high performance and quality.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = Cet64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C)]
pub struct Cet64 {
    s: u64,
}

const SP1: u64 = 0xFFFFFFFFFFFFFF43;
const SP2: u64 = 0xFFFFFFFFFFFFFF1B;
const P1: u64 = 0x94D049BB133111EB;

impl Cet64 {
    /// Creates a new `Cet64` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self { s: seedgen.nextu() }
    }
}

impl Rng64 for Cet64 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        self.s = self.s.wrapping_add(SP1);

        let mut x = self.s;
        x ^= x >> 30;
        x = x.wrapping_mul(SP2);
        x ^= x >> 27;
        x = x.wrapping_mul(P1);
        x ^= x >> 31;

        x
    }
}

pub struct Cet256 {
    s: [u64; 4],
}

impl Cet256 {
    /// Creates a new `Cet256` instance with a given seed.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            s: [
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
                seedgen.nextu(),
            ],
        }
    }
}

impl Rng64 for Cet256 {
    #[inline(always)]
    fn nextu(&mut self) -> u64 {
        self.s[0] = self.s[0].wrapping_add(SP1);
        let c0 = (self.s[0] < SP1) as u64;
        self.s[1] = self.s[1].wrapping_add(c0);
        let c1 = (self.s[1] < c0) as u64;
        self.s[2] = self.s[2].wrapping_add(c1);
        let c2 = (self.s[2] < c1) as u64;
        self.s[3] = self.s[3].wrapping_add(c2);

        let mut x = self.s[0] ^ self.s[3];
        x = x.wrapping_add(self.s[1].rotate_left(17));

        x ^= x >> 30;
        x = x.wrapping_mul(SP2);
        x ^= x >> 27;
        x = x.wrapping_mul(P1);
        x ^= x >> 31;

        x
    }
}

/// An 8-way SIMD CET64 generator using AVX-512 512-bit intrinsics.
#[repr(C, align(64))]
pub struct Cet64x8 {
    s: __m512i,
}

impl Cet64x8 {
    /// Creates a new `Cet64x8` from 8 independent seeds.
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 8];
        let mut seedgen = SplitMix64::new(seed);
        for lane in &mut s {
            *lane = seedgen.nextu();
        }
        unsafe {
            Self {
                s: _mm512_loadu_si512(s.as_ptr() as *const __m512i),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn mul_sp2_vec(x: __m512i) -> __m512i {
        let x8 = _mm512_slli_epi64(x, 8);
        let x5 = _mm512_slli_epi64(x, 5);
        let x2 = _mm512_slli_epi64(x, 2);
        let t0 = _mm512_sub_epi64(x8, x5);
        let t1 = _mm512_add_epi64(x2, x);
        let t = _mm512_add_epi64(t0, t1);
        _mm512_sub_epi64(_mm512_setzero_si512(), t)
    }

    /// Generates 8 random `u64` values simultaneously.
    #[target_feature(enable = "avx512f,avx512dq")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn nextu(&mut self) -> [u64; 8] {
        let sp1 = _mm512_set1_epi64(SP1 as i64);
        let p1 = _mm512_set1_epi64(P1 as i64);

        self.s = _mm512_add_epi64(self.s, sp1);

        let mut x = self.s;
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
        x = Self::mul_sp2_vec(x);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
        x = _mm512_mullo_epi64(x, p1);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));

        let mut out = [0u64; 8];
        _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, x);
        out
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn nextf(&mut self) -> [f64; 8] {
        let u = unsafe { self.nextu() };
        let scale = 1.0 / (u64::MAX as f64 + 1.0);
        let mut out = [0f64; 8];
        for i in 0..8 {
            out[i] = u[i] as f64 * scale;
        }
        out
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn randi(&mut self, min: i64, max: i64) -> [i64; 8] {
        let u = unsafe { self.nextu() };
        let range = (max as i128 - min as i128 + 1) as u128;
        let mut out = [0i64; 8];
        for i in 0..8 {
            out[i] = ((u[i] as u128 * range) >> 64) as i64 + min;
        }
        out
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn randf(&mut self, min: f64, max: f64) -> [f64; 8] {
        let u = unsafe { self.nextu() };
        let scale = (max - min) * (1.0 / (u64::MAX as f64 + 1.0));
        let mut out = [0f64; 8];
        for i in 0..8 {
            out[i] = u[i] as f64 * scale + min;
        }
        out
    }
}

/// A 2-way SIMD CET256 generator using AVX-512 storage layout.
///
/// Internal lane mapping is `[s0, s1, s2, s3, s0, s1, s2, s3]`.
#[repr(C, align(64))]
pub struct Cet256x2 {
    s: __m512i,
}

impl Cet256x2 {
    /// Creates a new `Cet256x2` from 2 independent seeds.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let mut s = [0u64; 8];
        for lane in &mut s {
            *lane = seedgen.nextu();
        }
        unsafe {
            Self {
                s: _mm512_loadu_si512(s.as_ptr() as *const __m512i),
            }
        }
    }

    /// Generates 2 random `u64` values simultaneously.
    #[target_feature(enable = "avx512f,avx512dq")]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn nextu(&mut self) -> [u64; 2] {
        let mut state = [0u64; 8];
        _mm512_storeu_si512(state.as_mut_ptr() as *mut __m512i, self.s);

        let mut lanes = [0u64; 8];
        for base in [0usize, 4usize] {
            state[base] = state[base].wrapping_add(SP1);
            let c0 = (state[base] < SP1) as u64;
            state[base + 1] = state[base + 1].wrapping_add(c0);
            let c1 = (state[base + 1] < c0) as u64;
            state[base + 2] = state[base + 2].wrapping_add(c1);
            let c2 = (state[base + 2] < c1) as u64;
            state[base + 3] = state[base + 3].wrapping_add(c2);

            lanes[base] =
                (state[base] ^ state[base + 3]).wrapping_add(state[base + 1].rotate_left(17));
        }

        self.s = _mm512_loadu_si512(state.as_ptr() as *const __m512i);

        let p1 = _mm512_set1_epi64(P1 as i64);
        let mut x = _mm512_loadu_si512(lanes.as_ptr() as *const __m512i);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
        x = Cet64x8::mul_sp2_vec(x);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
        x = _mm512_mullo_epi64(x, p1);
        x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));

        let mut out_lanes = [0u64; 8];
        _mm512_storeu_si512(out_lanes.as_mut_ptr() as *mut __m512i, x);
        [out_lanes[0], out_lanes[4]]
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn nextf(&mut self) -> [f64; 2] {
        let u = unsafe { self.nextu() };
        let scale = 1.0 / (u64::MAX as f64 + 1.0);
        [u[0] as f64 * scale, u[1] as f64 * scale]
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn randi(&mut self, min: i64, max: i64) -> [i64; 2] {
        let u = unsafe { self.nextu() };
        let range = (max as i128 - min as i128 + 1) as u128;
        [
            ((u[0] as u128 * range) >> 64) as i64 + min,
            ((u[1] as u128 * range) >> 64) as i64 + min,
        ]
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    pub unsafe fn randf(&mut self, min: f64, max: f64) -> [f64; 2] {
        let u = unsafe { self.nextu() };
        let scale = (max - min) * (1.0 / (u64::MAX as f64 + 1.0));
        [u[0] as f64 * scale + min, u[1] as f64 * scale + min]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Cet64);
    crate::safe_test!(Cet256);
    crate::unsafe_test!(Cet64x8);
    crate::unsafe_test!(Cet256x2);
}
