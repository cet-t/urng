use crate::rng::Rng64;

// --- TwistedGFSR ---

/// A Twisted Generalized Feedback Shift Register (TGFSR) generator.
#[repr(C, align(64))]
pub struct TwistedGFSR {
    seed: [u64; N_GFSR],
    index: usize,
}

const N_GFSR: usize = 25;
const M_GFSR: usize = 7;

impl TwistedGFSR {
    /// Provides a default seed array.
    pub const fn new_seed() -> [u64; N_GFSR] {
        [
            0x95f24dab, 0x0b685215, 0xe76ccae7, 0xaf3ec239, 0x715fad23, 0x24a590ad, 0x69e4b5ef,
            0xbf456141, 0x96bc1b7b, 0xa7bdf825, 0xc1de75b7, 0x8858a9c9, 0x2da87693, 0xb657f9dd,
            0xffdc8a9f, 0x8121da71, 0x8b823ecb, 0x885d05f5, 0x4e20cd47, 0x5a9ad5d9, 0x512c0c03,
            0xea857ccd, 0x4cc1d30f, 0x8891a8a1, 0xa6b7aadb,
        ]
    }
    const fn mag01() -> [u64; 2] {
        [0x0, 0x8ebfd028]
    }

    /// Creates a new `TwistedGFSR` instance.
    pub fn new(seed: [u64; N_GFSR]) -> Self {
        Self {
            seed,
            index: N_GFSR,
        }
    }

    fn twist(&mut self) {
        for k in 0..(N_GFSR - M_GFSR) {
            self.seed[k] = self.seed[k + M_GFSR]
                ^ (self.seed[k] >> 1)
                ^ Self::mag01()[(self.seed[k] & 1) as usize];
        }
        for k in (N_GFSR - M_GFSR)..N_GFSR {
            self.seed[k] = self.seed[k + M_GFSR - N_GFSR]
                ^ (self.seed[k] >> 1)
                ^ Self::mag01()[(self.seed[k] & 1) as usize];
        }
        self.index = 0;
    }

    /// Generates the next random `u32` value.
    #[inline]
    pub fn nextu(&mut self) -> u64 {
        if self.index >= N_GFSR {
            self.twist();
        }
        let mut y = self.seed[self.index];
        y ^= (y << 7) & 0x2b5b2500;
        y ^= (y << 15) & 0xdb8b0000;
        y &= 0xffffffff;
        y ^= y >> 16;
        self.index += 1;
        y
    }

    /// Generates the next random `f64` value in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> f64 {
        self.nextu() as f64 * (1.0 / 4294967296.0)
    }

    /// Generates a random `i64` value in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> i64 {
        let range = (max as i128 - min as i128 + 1) as u128;
        let x = self.nextu();
        ((x as u128 * range) >> 64) as i64 + min
    }

    /// Generates a random `f64` value in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> f64 {
        let range = max - min;
        let scale = range * (1.0 / (u64::MAX as f64 + 1.0));
        (self.nextu() as f64 * scale) + min
    }

    /// Returns a random element from a slice.
    #[inline]
    pub fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.randi(0, choices.len() as i64 - 1);
        &choices[index as usize]
    }
}

impl Rng64 for TwistedGFSR {
    #[inline]
    fn randi(&mut self, min: i64, max: i64) -> i64 {
        self.randi(min, max)
    }
    #[inline]
    fn randf(&mut self, min: f64, max: f64) -> f64 {
        self.randf(min, max)
    }
    #[inline]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        self.choice(choices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twisted_gfsr_works() {
        let mut rng = TwistedGFSR::new(TwistedGFSR::new_seed());
        assert_eq!(rng.nextu(), 868393086);
        assert_eq!(rng.nextf(), 0.33567164628766477);
    }
}
