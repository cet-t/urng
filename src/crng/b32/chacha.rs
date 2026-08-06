use wrapn::{wrap, wu32, wusize};

use crate::{Rng, SplitMix32, impl_ring_rng32};

pub struct ChaCha<const ROUNDS: usize> {
    x: [wu32; 16],

    pub(crate) buf: [wu32; 16],
    pub(crate) pos: wusize,
}

impl<const ROUNDS: usize> ChaCha<ROUNDS> {
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
        a += b;
        d ^= a;
        d = d.rotate_left(16);

        c += d;
        b ^= c;
        b = b.rotate_left(12);

        a += b;
        d ^= a;
        d = d.rotate_left(8);

        c += d;
        b ^= c;
        b = b.rotate_left(7);

        [a, b, c, d]
    }

    fn next_raw(&mut self) -> [u32; 16] {
        let mut x = self.x;
        for _ in 0..ROUNDS {
            [x[0], x[4], x[8], x[12]] = Self::qr(x[0], x[4], x[8], x[12]);
            [x[1], x[5], x[9], x[13]] = Self::qr(x[1], x[5], x[9], x[13]);
            [x[2], x[6], x[10], x[14]] = Self::qr(x[2], x[6], x[10], x[14]);
            [x[3], x[7], x[11], x[15]] = Self::qr(x[3], x[7], x[11], x[15]);

            [x[0], x[5], x[10], x[15]] = Self::qr(x[0], x[5], x[10], x[15]);
            [x[1], x[6], x[11], x[12]] = Self::qr(x[1], x[6], x[11], x[12]);
            [x[2], x[7], x[8], x[13]] = Self::qr(x[2], x[7], x[8], x[13]);
            [x[3], x[4], x[9], x[14]] = Self::qr(x[3], x[4], x[9], x[14]);
        }

        self.x = x;
        x.map(|x| x.value())
    }
}

impl<const ROUNDS: usize> Default for ChaCha<ROUNDS> {
    fn default() -> Self {
        Self::new(crate::_internal::default_seed32())
    }
}

impl<const ROUNDS: usize> crate::Seed for ChaCha<ROUNDS> {
    type Seed = u32;

    #[inline]
    fn from_seed(seed: Self::Seed) -> Self {
        Self::new(seed)
    }
}

pub type ChaCha8 = ChaCha<8>;
impl_ring_rng32!(ChaCha8, 16, next_raw);

pub type ChaCha20 = ChaCha<20>;
impl_ring_rng32!(ChaCha20, 16, next_raw);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe_test;

    safe_test!(ChaCha8);
    safe_test!(ChaCha20);
}
