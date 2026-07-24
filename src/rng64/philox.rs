use wrapn::{Wrap, wrap};

use crate::_internal::{impl_ring_rng64, impl_seed};
#[allow(unused_imports)]
use crate::{_internal::FSCALE64, rng::Rng, rng64::SplitMix64};

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
/// use urng::rng64::Philox64;
/// use urng::rng::Rng;
///
/// let mut rng = Philox64::new(1);
/// let _: u64 = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Philox64 {
    pub(crate) c: [Wrap<u64>; 2],
    pub(crate) k: [Wrap<u64>; 2],
    pub(crate) buf: [Wrap<u64>; 2],
    pub(crate) pos: Wrap<usize>,
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

    /// Generates the next block of 2 random `u64` values in one call.
    ///
    /// This is the raw bulk-generation path (used internally to refill the
    /// scalar [`Rng::nextu`] buffer, and available directly for
    /// throughput-sensitive callers that want the whole block at once).
    #[inline]
    pub fn next_raw(&mut self) -> [u64; 2] {
        let out = Self::compute(self.c.map(|x| x.value()), self.k.map(|x| x.value()));
        self.c[0] += 1;
        if self.c[0] == 0 {
            self.c[1] += 1;
        }
        out
    }
}

impl_seed!(Philox64, 64);

impl_ring_rng64!(Philox64, 2, next_raw);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Philox64);
}
