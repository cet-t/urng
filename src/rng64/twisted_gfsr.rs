use crate::{rng::Rng64, rng64::SplitMix64};

// --- TwistedGFSR ---

/// A Twisted Generalized Feedback Shift Register (TGFSR) generator.
///
/// # Examples
///
/// ```
/// use urng::prelude::*;
///
/// let mut rng = TwistedGFSR::new(0);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct TwistedGFSR {
    seed: [u32; N_GFSR],
    index: usize,
}

const N_GFSR: usize = 25;
const M_GFSR: usize = 7;
// const SEED: [u64; N_GFSR] = [
//     0x95f24dab, 0x0b685215, 0xe76ccae7, 0xaf3ec239, 0x715fad23, 0x24a590ad, 0x69e4b5ef, 0xbf456141,
//     0x96bc1b7b, 0xa7bdf825, 0xc1de75b7, 0x8858a9c9, 0x2da87693, 0xb657f9dd, 0xffdc8a9f, 0x8121da71,
//     0x8b823ecb, 0x885d05f5, 0x4e20cd47, 0x5a9ad5d9, 0x512c0c03, 0xea857ccd, 0x4cc1d30f, 0x8891a8a1,
//     0xa6b7aadb,
// ];
const MAG01: [u32; 2] = [0x0, 0x8ebf_d028];

impl TwistedGFSR {
    /// Creates a new `TwistedGFSR` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            seed: [0u32; N_GFSR].map(|_| seedgen.nextu() as u32),
            index: N_GFSR,
        }
    }

    fn twist(&mut self) {
        for k in 0..(N_GFSR - M_GFSR) {
            self.seed[k] =
                self.seed[k + M_GFSR] ^ (self.seed[k] >> 1) ^ MAG01[(self.seed[k] & 1) as usize];
        }
        for k in (N_GFSR - M_GFSR)..N_GFSR {
            self.seed[k] = self.seed[k + M_GFSR - N_GFSR]
                ^ (self.seed[k] >> 1)
                ^ MAG01[(self.seed[k] & 1) as usize];
        }
        self.index = 0;
    }
}

impl Rng64 for TwistedGFSR {
    #[inline]
    fn nextu(&mut self) -> u64 {
        if self.index >= N_GFSR {
            self.twist();
        }
        let mut y = self.seed[self.index];
        y ^= (y << 7) & 0x2b5b_2500;
        y ^= (y << 15) & 0xdb8b_0000;
        y ^= y >> 16;
        self.index += 1;
        u64::from(y)
    }

    #[inline(always)]
    fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0))
    }

    #[inline(always)]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        ((self.nextu() as u128 * range) >> 32) as i64 + min
    }

    #[inline(always)]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        min + self.nextf() * (max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(TwistedGFSR);
}
