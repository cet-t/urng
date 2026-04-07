use crate::{rng::Rng64, rng64::SplitMix64};

// --- Threefish256 ---

const THREEFISH_C240: u64 = 0x1BD11BDAA9FC1A22;
const THREE_FISH_N_ROUNDS: usize = 72;
const THREEFISH_PI: [usize; 4] = [0, 3, 2, 1];

// key_schedule は s=0..=18 で呼ばれる。% 5 / % 3 をコンパイル時テーブルで排除。
const KS_N: usize = THREE_FISH_N_ROUNDS / 4 + 1; // 19
const KS_K_IDX: [[usize; 4]; KS_N] = {
    let mut t = [[0usize; 4]; KS_N];
    let mut s = 0;
    while s < KS_N {
        t[s] = [s % 5, (s + 1) % 5, (s + 2) % 5, (s + 3) % 5];
        s += 1;
    }
    t
};
const KS_TW_IDX: [[usize; 2]; KS_N] = {
    let mut t = [[0usize; 2]; KS_N];
    let mut s = 0;
    while s < KS_N {
        t[s] = [s % 3, (s + 1) % 3];
        s += 1;
    }
    t
};

const THREEFISH_R_256: [[u32; 2]; 8] = [
    [14, 16],
    [52, 57],
    [23, 40],
    [5, 37],
    [25, 33],
    [46, 12],
    [58, 22],
    [32, 32],
];

/// A Threefish-256 random number generator.
///
/// # Examples
///
/// ```
/// use urng::rng64::Threefish256;
///
/// let mut rng = Threefish256::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Threefish256 {
    c: [u64; 4],
    k: [u64; 5],
    tw: [u64; 3],
    index: usize,
    buffer: [u64; 4],
}

impl Threefish256 {
    /// Creates a new `Threefish256` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        let mut k = [0u64; 5];
        k[0] = seedgen.nextu();
        k[1] = seedgen.nextu();
        k[2] = seedgen.nextu();
        k[3] = seedgen.nextu();
        k[4] = THREEFISH_C240 ^ k[0] ^ k[1] ^ k[2] ^ k[3];

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

    #[inline(always)]
    fn mix(x0: u64, x1: u64, r: u32) -> [u64; 2] {
        let y0 = x0.wrapping_add(x1);
        [y0, x1.rotate_left(r) ^ y0]
    }

    #[inline(always)]
    fn key_schedule(k: &[u64; 5], tw: &[u64; 3], s: usize) -> [u64; 4] {
        let ki = KS_K_IDX[s];
        let ti = KS_TW_IDX[s];
        [
            k[ki[0]],
            k[ki[1]].wrapping_add(tw[ti[0]]),
            k[ki[2]].wrapping_add(tw[ti[1]]),
            k[ki[3]].wrapping_add(s as u64),
        ]
    }

    #[inline(always)]
    fn next_block(&mut self) -> [u64; 4] {
        let mut v = self.c;

        for r in 0..THREE_FISH_N_ROUNDS {
            let mut e = [0u64; 4];
            if (r & 0b011) == 0 {
                let ksi = Self::key_schedule(&self.k, &self.tw, r >> 2);
                e[0] = v[0].wrapping_add(ksi[0]);
                e[1] = v[1].wrapping_add(ksi[1]);
                e[2] = v[2].wrapping_add(ksi[2]);
                e[3] = v[3].wrapping_add(ksi[3]);
            } else {
                e = v;
            }

            let mut f = [0u64; 4];
            let r_sh = THREEFISH_R_256[r & 0b0111]; // r % 8
            let mx0 = Self::mix(e[0], e[1], r_sh[0]);
            f[0] = mx0[0];
            f[1] = mx0[1];
            let mx1 = Self::mix(e[2], e[3], r_sh[1]);
            f[2] = mx1[0];
            f[3] = mx1[1];

            v[0] = f[THREEFISH_PI[0]];
            v[1] = f[THREEFISH_PI[1]];
            v[2] = f[THREEFISH_PI[2]];
            v[3] = f[THREEFISH_PI[3]];
        }

        let ksi = Self::key_schedule(&self.k, &self.tw, THREE_FISH_N_ROUNDS.div_ceil(4));
        let mut dst = [0u64; 4];
        for i in 0..4 {
            dst[i] = v[i].wrapping_add(ksi[i]) ^ self.c[i];
        }

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

    /// Generates the next random `u64` values.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 4] {
        if self.index >= 4 {
            self.buffer = self.next_block();
            self.index = 0;
        }
        let val = self.buffer;
        self.index += 4;
        val
    }

    /// Generates the next random `f64` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f64; 4] {
        let out = self.nextu();
        const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
        let mut dst = [0f64; 4];
        dst[0] = out[0] as f64 * SCALE;
        dst[1] = out[1] as f64 * SCALE;
        dst[2] = out[2] as f64 * SCALE;
        dst[3] = out[3] as f64 * SCALE;
        dst
    }

    /// Generates random `i64` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> [i64; 4] {
        let range = (max as i128 - min as i128 + 1) as u128;
        let out = self.nextu();
        let dst = [
            ((out[0] as u128 * range) >> 64) as i64 + min,
            ((out[1] as u128 * range) >> 64) as i64 + min,
            ((out[2] as u128 * range) >> 64) as i64 + min,
            ((out[3] as u128 * range) >> 64) as i64 + min,
        ];
        dst
    }

    /// Generates random `f64` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> [f64; 4] {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        let out = self.nextu();
        let dst = [
            (out[0] as f64 * scale) + min,
            (out[1] as f64 * scale) + min,
            (out[2] as f64 * scale) + min,
            (out[3] as f64 * scale) + min,
        ];
        dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threefish256_works() {
        let mut rng = Threefish256::new(1);
        assert_eq!(
            rng.nextu(),
            [
                11703024954964515355,
                12040493508789228569,
                15247998991077977543,
                2489860152104538722
            ]
        );
        assert_eq!(
            rng.nextf(),
            [
                0.9622286493050641,
                0.5334118826690859,
                0.5452741654192154,
                0.6320415850102533
            ]
        );
    }
}
