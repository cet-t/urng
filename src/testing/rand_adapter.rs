//! Adapter allowing `rand`-ecosystem RNGs (`rand_core::RngCore`) to be used with
//! urng's testing harness ([`ChiSq32`](crate::testing::ChiSq32), [`McPi32`](crate::testing::McPi32), etc.),
//! which are otherwise generic over urng's own [`Rng32`]/[`Rng64`] traits.

use crate::rng::{Rng32, Rng64};
use rand_core::Rng as RandRng;

/// Wraps any `rand_core::Rng` implementation so it satisfies urng's [`Rng32`]/[`Rng64`] traits.
pub struct RandAdapter<R: RandRng>(pub R);

impl<R: RandRng> RandAdapter<R> {
    pub fn new(rng: R) -> Self {
        Self(rng)
    }
}

impl<R: RandRng> Rng32 for RandAdapter<R> {
    fn nextu(&mut self) -> u32 {
        self.0.next_u32()
    }
}

impl<R: RandRng> Rng64 for RandAdapter<R> {
    fn nextu(&mut self) -> u64 {
        self.0.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ChiSq32, ChiSqVerdict, McPi32, McPiVerdict};
    use std::convert::Infallible;

    /// Minimal RNG implementing `rand_core::TryRng` directly, standing in for
    /// any external `rand`-ecosystem generator (which all implement this trait).
    struct DummyRandRng(u32);

    impl rand_core::TryRng for DummyRandRng {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            self.0 = self.0.wrapping_add(0x9e3779b9);
            let mut z = self.0;
            z = (z ^ (z >> 16)).wrapping_mul(0x85ebca6b);
            z = (z ^ (z >> 13)).wrapping_mul(0xc2b2ae35);
            Ok(z ^ (z >> 16))
        }

        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            let hi = self.try_next_u32()? as u64;
            let lo = self.try_next_u32()? as u64;
            Ok(hi << 32 | lo)
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
            let mut i = 0;
            while i < dst.len() {
                let val = self.try_next_u32()?;
                let take = (dst.len() - i).min(4);
                dst[i..i + take].copy_from_slice(&val.to_le_bytes()[..take]);
                i += take;
            }
            Ok(())
        }
    }

    #[test]
    fn rand_adapter_is_deterministic() {
        let mut a = RandAdapter::new(DummyRandRng(1));
        let mut b = RandAdapter::new(DummyRandRng(1));
        assert_eq!(Rng32::nextu(&mut a), Rng32::nextu(&mut b));
        assert_eq!(Rng64::nextu(&mut a), Rng64::nextu(&mut b));
    }

    #[test]
    fn rand_adapter_works_with_chisq() {
        let mut rng = RandAdapter::new(DummyRandRng(42));
        let mut chisq = ChiSq32::new(&mut rng);
        let res = chisq.run("dummy_rand_rng").unwrap();
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }

    #[test]
    fn rand_adapter_works_with_mcpi() {
        let mut rng = RandAdapter::new(DummyRandRng(7));
        let mut mcpi = McPi32::new(&mut rng);
        let res = mcpi.run("dummy_rand_rng").unwrap();
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }
}
