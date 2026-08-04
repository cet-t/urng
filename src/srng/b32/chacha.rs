use wrapn::{Wrap, wrap};

use crate::{Rng, SplitMix32, impl_ring_rng32};

const ROUNDS: usize = 20;

pub struct ChaCha20 {
    x: [Wrap<u32>; 16],

    pub(crate) buf: [Wrap<u32>; 16],
    pub(crate) pos: Wrap<usize>,
}

impl ChaCha20 {
    pub fn new(seed: u32) -> Self {
        let mut sg = SplitMix32::new(seed);
        let x = [0_u32; 16].map(|_| wrap!(sg.nextu()));
        Self {
            x,
            buf: wrap![0_u32; 16],
            pos: wrap!(16),
        }
    }

    fn qr(
        mut a: Wrap<u32>,
        mut b: Wrap<u32>,
        mut c: Wrap<u32>,
        mut d: Wrap<u32>,
    ) -> [Wrap<u32>; 4] {
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
            [x[00], x[04], x[08], x[12]] = Self::qr(x[00], x[04], x[08], x[12]);
            [x[01], x[05], x[09], x[13]] = Self::qr(x[01], x[05], x[09], x[13]);
            [x[02], x[06], x[10], x[14]] = Self::qr(x[02], x[06], x[10], x[14]);
            [x[03], x[07], x[11], x[15]] = Self::qr(x[03], x[07], x[11], x[15]);

            [x[00], x[05], x[10], x[15]] = Self::qr(x[00], x[05], x[10], x[15]);
            [x[01], x[06], x[11], x[12]] = Self::qr(x[01], x[06], x[11], x[12]);
            [x[02], x[07], x[08], x[13]] = Self::qr(x[02], x[07], x[08], x[13]);
            [x[03], x[04], x[09], x[14]] = Self::qr(x[03], x[04], x[09], x[14]);
        }
        x.map(|x| x.value())
    }
}

impl_ring_rng32!(ChaCha20, 16, next_raw);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe_test;

    safe_test!(ChaCha20);
}
