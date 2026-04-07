use crate::{rng::Rng32, rng32::SplitMix32};

// --- Threefry32x4 ---

const THREEFRY32_C240: u32 = 0x1BD11BDA;
/// A Threefry4x32 random number generator (Random123 family).
///
/// This is a counter-based RNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 4 output values per block.
///
/// # Examples
///
/// ```
/// use urng::rng32::Threefry32x4;
///
/// let mut rng = Threefry32x4::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Threefry32x4 {
    pub(crate) c: [u32; 4],
    pub(crate) k: [u32; 5],
    pub(crate) tw: [u32; 3],
    pub(crate) index: usize,
    pub(crate) buffer: [u32; 4],
}

impl Threefry32x4 {
    /// Creates a new `Threefry32x4` instance seeded with the given value.
    ///
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut k = [0u32; 5];
        for i in 0..4 {
            k[i] = seedgen.nextu();
        }
        k[4] = THREEFRY32_C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3];

        let tw0 = seedgen.nextu();
        let tw1 = seedgen.nextu();
        let tw = [tw0, tw1, tw0 ^ tw1];

        Self {
            c: [0; 4],
            k,
            tw,
            index: 4,
            buffer: [0; 4],
        }
    }

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 4 output words.
    ///
    /// # Arguments
    ///
    /// * `c`  - 4-word counter (the plaintext block).
    /// * `k`  - 5-word key schedule (k\[4\] = parity word).
    /// * `tw` - 3-word tweak schedule (tw\[2\] = tw\[0\] ^ tw\[1\]).
    #[inline(always)]
    pub fn compute(c: [u32; 4], k: &[u32; 5], tw: &[u32; 3]) -> [u32; 4] {
        let mut v = c;

        macro_rules! round {
            ($r_sh_0:expr, $r_sh_1:expr) => {
                let y0 = v[0].wrapping_add(v[1]);
                let f1 = v[1].rotate_left($r_sh_0) ^ y0;

                let y1 = v[2].wrapping_add(v[3]);
                let f3 = v[3].rotate_left($r_sh_1) ^ y1;

                v[0] = y0;
                v[1] = f3;
                v[2] = y1;
                v[3] = f1;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] = v[0].wrapping_add(k[$s % 5]);
                v[1] = v[1].wrapping_add(k[($s + 1) % 5].wrapping_add(tw[$s % 3]));
                v[2] = v[2].wrapping_add(k[($s + 2) % 5].wrapping_add(tw[($s + 1) % 3]));
                v[3] = v[3].wrapping_add(k[($s + 3) % 5].wrapping_add($s as u32));
            };
        }

        inject_key!(0);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(1);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(2);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(3);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(4);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        let ksi5_0 = k[0];
        let ksi5_1 = k[1].wrapping_add(tw[2]);
        let ksi5_2 = k[2].wrapping_add(tw[0]);
        let ksi5_3 = k[3].wrapping_add(5);

        [
            v[0].wrapping_add(ksi5_0) ^ c[0],
            v[1].wrapping_add(ksi5_1) ^ c[1],
            v[2].wrapping_add(ksi5_2) ^ c[2],
            v[3].wrapping_add(ksi5_3) ^ c[3],
        ]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u32; 4] {
        let dst = Self::compute(self.c, &self.k, &self.tw);

        self.c[0] = self.c[0].wrapping_add(1);
        if self.c[0] == 0 {
            self.c[1] = self.c[1].wrapping_add(1);
            if self.c[1] == 0 {
                self.c[2] = self.c[2].wrapping_add(1);
                if self.c[2] == 0 {
                    self.c[3] = self.c[3].wrapping_add(1);
                }
            }
        }

        dst
    }

    /// Generates the next 4 random `u32` values.
    #[inline]
    pub fn nextu(&mut self) -> [u32; 4] {
        if self.index >= 4 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 4;
        val
    }

    /// Generates the next 4 random `f32` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f32; 4] {
        let out = self.nextu();
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
        let mut dst = [0f32; 4];
        for i in 0..4 {
            dst[i] = out[i] as f32 * SCALE;
        }
        dst
    }

    /// Generates 4 random `i32` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> [i32; 4] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = self.nextu();
        let mut dst = [0i32; 4];
        for i in 0..4 {
            dst[i] = ((out[i] as u64 * range) >> 32) as i32 + min;
        }
        dst
    }

    /// Generates 4 random `f32` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> [f32; 4] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = self.nextu();
        let mut dst = [0f32; 4];
        for i in 0..4 {
            dst[i] = (out[i] as f32 * scale) + min;
        }
        dst
    }
}

// --- Threefry32x2 ---

/// A Threefry 2x32 random number generator (Random123 family).
///
/// Counter-based PRNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 2 output values per block.
///
/// # Examples
///
/// ```
/// use urng::rng32::Threefry32x2;
///
/// let mut rng = Threefry32x2::new(1);
/// let _ = rng.nextu();
/// ```
pub struct Threefry32x2 {
    pub(crate) c: [u32; 2],
    pub(crate) k: [u32; 3],
    pub(crate) buffer: [u32; 2],
    pub(crate) index: usize,
}

impl Threefry32x2 {
    /// Creates a new `Threefry32x2` instance seeded with the given value.
    ///
    #[inline]
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        let k0 = sm.nextu();
        let k1 = sm.nextu();
        let k2 = k0 ^ k1 ^ THREEFRY32_C240;

        Self {
            c: [0, 0],
            k: [k0, k1, k2],
            buffer: [0; 2],
            index: 2,
        }
    }

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 2 output words.
    ///
    /// # Arguments
    ///
    /// * `c` - 2-word counter (the plaintext block).
    /// * `k` - 3-word key schedule (k[2] = k[0] ^ k[1] ^ C240).
    #[inline(always)]
    pub(crate) fn compute(c: [u32; 2], k: &[u32; 3]) -> [u32; 2] {
        let mut v = c;

        macro_rules! round {
            ($r_sh:expr) => {
                let y = v[0].wrapping_add(v[1]);
                let f = v[1].rotate_left($r_sh) ^ y;
                v[0] = y;
                v[1] = f;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] = v[0].wrapping_add(k[$s % 3]);
                v[1] = v[1].wrapping_add(k[($s + 1) % 3].wrapping_add($s as u32));
            };
        }

        inject_key!(0);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(1);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(2);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(3);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(4);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        let ksi5_0 = k[2];
        let ksi5_1 = k[0].wrapping_add(5);

        [v[0].wrapping_add(ksi5_0), v[1].wrapping_add(ksi5_1)]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u32; 2] {
        let dst = Self::compute(self.c, &self.k);
        let (n_c0, overflow) = self.c[0].overflowing_add(1);
        self.c[0] = n_c0;
        if overflow {
            self.c[1] = self.c[1].wrapping_add(1);
        }
        dst
    }

    /// Generates the next 2 random `u32` values.
    #[inline]
    pub fn nextu(&mut self) -> [u32; 2] {
        if self.index >= 2 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 2;
        val
    }

    /// Generates the next 2 random `f32` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f32; 2] {
        let out = self.nextu();
        const SCALE: f32 = 1.0 / (u32::MAX as f32 + 1.0);
        [out[0] as f32 * SCALE, out[1] as f32 * SCALE]
    }

    /// Generates 2 random `i32` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i32, max: i32) -> [i32; 2] {
        let range = (max as i64 - min as i64 + 1) as u64;
        let out = self.nextu();
        [
            ((out[0] as u64 * range) >> 32) as i32 + min,
            ((out[1] as u64 * range) >> 32) as i32 + min,
        ]
    }

    /// Generates 2 random `f32` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f32, max: f32) -> [f32; 2] {
        let range = max - min;
        let scale = range * (1.0 / (u32::MAX as f32 + 1.0));
        let out = self.nextu();
        [(out[0] as f32 * scale) + min, (out[1] as f32 * scale) + min]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefry32x4_works() {
        let mut rng = Threefry32x4::new(1);
        assert_eq!(rng.nextu(), [215661891, 4046822497, 3522917133, 3418596171]);
        assert_eq!(
            rng.nextf(),
            [0.05775363, 0.54074997, 0.15642758, 0.23995495]
        );
    }

    #[test]
    fn threefry32x2_works() {
        let mut rng = Threefry32x2::new(1);
        assert_eq!(rng.nextu(), [3732229352, 2044399418]);
        assert_eq!(rng.nextf(), [0.092225075, 0.077477075]);
    }
}
