use wrapn::{wrap, wu32, wusize};

use crate::{Rng, SplitMix32, impl_ring_rng32};

/// Salsa stream cipher based RNG implementation, generic over the round count.
///
/// # Example
/// ```
/// use urng::{Rng, Salsa20};
///
/// let mut rng = Salsa20::new(12345);
/// let _ = rng.nextu();
/// ```
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Salsa<const ROUNDS: usize> {
    x: [wu32; 16],

    pub(crate) buf: [wu32; 16],
    pub(crate) pos: wusize,
}

impl<const ROUNDS: usize> Salsa<ROUNDS> {
    /// Creates a new `Salsa` instance with the given seed.
    pub fn new(seed: u32) -> Self {
        let mut sg = SplitMix32::new(seed);

        let x = [0_u32; 16].map(|_| wrap!(sg.nextu()));
        Self {
            x,

            buf: wrap![0_u32; 16],
            pos: wrap!(16),
        }
    }

    fn qr(mut a: wu32, mut b: wu32, mut c: wu32, mut d: wu32) -> [wu32; 4] {
        b ^= (a + d).rotate_left(7);
        c ^= (b + a).rotate_left(9);
        d ^= (c + b).rotate_left(9);
        a ^= (d + c).rotate_left(18);
        [a, b, c, d]
    }

    fn next_raw(&mut self) -> [u32; 16] {
        let mut x = self.x;
        for _ in 0..ROUNDS {
            [x[0], x[4], x[8], x[12]] = Self::qr(x[0], x[4], x[8], x[12]);
            [x[5], x[9], x[13], x[1]] = Self::qr(x[5], x[9], x[13], x[1]);
            [x[10], x[14], x[2], x[6]] = Self::qr(x[10], x[14], x[2], x[6]);
            [x[15], x[3], x[7], x[11]] = Self::qr(x[15], x[3], x[7], x[11]);

            [x[0], x[1], x[2], x[3]] = Self::qr(x[0], x[1], x[2], x[3]);
            [x[5], x[6], x[7], x[4]] = Self::qr(x[5], x[6], x[7], x[4]);
            [x[10], x[11], x[8], x[9]] = Self::qr(x[10], x[11], x[8], x[9]);
            [x[15], x[12], x[13], x[14]] = Self::qr(x[15], x[12], x[13], x[14]);
        }
        self.x = x;
        x.map(|x| *x)
    }
}

impl<const ROUNDS: usize> Default for Salsa<ROUNDS> {
    fn default() -> Self {
        Self::new(crate::_internal::default_seed32())
    }
}

impl<const ROUNDS: usize> crate::Seed for Salsa<ROUNDS> {
    type Seed = u32;

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(seed)
    }
}

pub type Salsa8 = Salsa<8>;
impl_ring_rng32!(Salsa8, 16, next_raw);

pub type Salsa20 = Salsa<20>;
impl_ring_rng32!(Salsa20, 16, next_raw);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! {
        Salsa8,
        Salsa20
    }
}
