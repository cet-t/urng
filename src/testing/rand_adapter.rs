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
