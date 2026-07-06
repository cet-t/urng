use wrapn::{Wrap, wrap};

use crate::{_internal::FSCALE64, rng::Rng64, rng64::SplitMix64};

// --- Philox64 ---

/// A Philox 2x64 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications.
///
/// # Examples
///
/// ```
/// use urng::rng64::Philox64;
///
/// let mut rng = Philox64::new(1);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Philox64 {
    pub(crate) c: [Wrap<u64>; 2],
    pub(crate) k: [Wrap<u64>; 2],
}

impl Philox64 {
    /// Creates a new `Philox64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            c: wrap![1, 0],
            k: wrap![seedgen.nextu(), seedgen.nextu()],
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline]
    pub(crate) fn compute(c: [u64; 2], k: [u64; 2]) -> [u64; 2] {
        let mut v = wrap![c[0], c[1]];
        let mut key = k[0];

        const M0: u128 = 0xD2B74407B1CE6E93;
        const W0: u64 = 0x9E3779B97F4A7C15;

        macro_rules! step {
            () => {
                step!(fin);
                key = key.wrapping_add(W0);
            };
            (fin) => {
                let prod = v[0].cast::<u128>() * M0;
                let hi = (prod >> 64).cast::<u64>();
                let lo = prod.cast::<u64>();

                v[0] = hi ^ v[1] ^ key;
                v[1] = lo;
            };
        }

        step!();
        step!();
        step!();
        step!();
        step!();
        step!();
        step!();
        step!();
        step!();
        step!(fin);

        v.map(|x| x.value())
    }

    /// Generates the next block of random numbers.
    #[inline]
    pub fn nextu(&mut self) -> [u64; 2] {
        let out = Self::compute(self.c.map(|x| x.value()), self.k.map(|x| x.value()));
        self.c[0] += 1;
        if self.c[0] == 0 {
            self.c[1] += 1;
        }
        out
    }

    /// Generates a random `f64` values in the range [0, 1).
    #[inline]
    pub fn nextf(&mut self) -> [f64; 2] {
        self.nextu().map(|x| (x as f64) * FSCALE64)
    }

    /// Generates a random `i64` values in the range [min, max].
    #[inline]
    pub fn randi(&mut self, min: i64, max: i64) -> [i64; 2] {
        let range = (max as i128 - min as i128 + 1) as u128;
        self.nextu()
            .map(|x| ((x as u128 * range) >> 64) as i64 + min)
    }

    /// Generates a random `f64` values in the range [min, max).
    #[inline]
    pub fn randf(&mut self, min: f64, max: f64) -> [f64; 2] {
        let scale = (max - min) * FSCALE64;
        self.nextu().map(|x| (x as f64 * scale) + min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Philox64);
}
