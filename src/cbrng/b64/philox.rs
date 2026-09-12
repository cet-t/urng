use wrapn::{wrap, wu64, wusize};

use crate::_internal::impl_ring_rng64;
use crate::{prng::b64::SplitMix64, rng::Rng};

// --- Philox64 ---

/// A Philox 2x64 random number generator.
///
/// This is a counter-based RNG suitable for parallel applications. Implements
/// [`Rng`] directly: each call to [`Rng::nextu`] hands out one `u64` from
/// an internal 2-word buffer, recomputing a fresh block every 2nd call.
///
/// # Examples
///
/// ```
/// use urng::{Rng, Philox64};
///
/// let mut rng = Philox64::new(1);
/// let _: u64 = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Philox64 {
    pub(crate) c: [wu64; 2],
    pub(crate) k: [wu64; 2],
    pub(crate) buf: [wu64; 2],
    pub(crate) pos: wusize,
}

impl Philox64 {
    /// Creates a new `Philox64` instance.
    pub fn new(seed: u64) -> Self {
        let mut seedgen = SplitMix64::new(seed);
        Self {
            c: wrap![1, 0],
            k: wrap![seedgen.nextu(), seedgen.nextu()],
            buf: wrap![0; 2],
            pos: 2.into(),
        }
    }

    /// Computes Philox output from counter and key values (pure function).
    #[inline]
    pub(crate) fn compute(mut c: [wu64; 2], k: [wu64; 2]) -> [wu64; 2] {
        let mut key = k[0];

        const M0: u128 = 0xD2B74407B1CE6E93;
        const W0: u64 = 0x9E3779B97F4A7C15;

        macro_rules! step {
            () => {
                step!(fin);
                key += W0;
            };
            (fin) => {
                let prod = c[0].cast::<u128>() * M0;
                let hi = (prod >> 64).cast::<u64>();
                let lo = prod.cast::<u64>();

                c[0] = hi ^ c[1] ^ key;
                c[1] = lo;
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

        c
    }

    /// Generates the next block of 2 random `u64` values in one call.
    ///
    /// This is the raw bulk-generation path (used internally to refill the
    /// scalar [`Rng::nextu`] buffer, and available directly for
    /// throughput-sensitive callers that want the whole block at once).
    #[inline]
    pub fn next_raw(&mut self) -> [u64; 2] {
        let out = Self::compute(self.c, self.k);
        self.c[0] += 1;
        if self.c[0] == 0 {
            self.c[1] += 1;
        }
        out.map(|x| *x)
    }
}

impl_ring_rng64!(Philox64, 2, next_raw);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! { Philox64 }
}
