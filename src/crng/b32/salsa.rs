use wrapn::{Wrap, wrap};

use crate::{Rng, SplitMix32, impl_ring_rng32};

pub struct Salsa<const ROUNDS: usize> {
    x: [Wrap<u32>; 16],

    pub(crate) buf: [Wrap<u32>; 16],
    pub(crate) pos: Wrap<usize>,
}

impl<const ROUNDS: usize> Salsa<ROUNDS> {
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
        b ^= (a + d).rotate_left(7);
        c ^= (b + a).rotate_left(9);
        d ^= (c + b).rotate_left(9);
        a ^= (d + c).rotate_left(18);
        [a, b, c, d]
    }

    fn next_raw(&mut self) -> [u32; 16] {
        let mut x = self.x;
        for _ in 0..ROUNDS {
            [x[00], x[04], x[08], x[12]] = Self::qr(x[00], x[04], x[08], x[12]);
            [x[05], x[09], x[13], x[01]] = Self::qr(x[05], x[09], x[13], x[01]);
            [x[10], x[14], x[02], x[06]] = Self::qr(x[10], x[14], x[02], x[06]);
            [x[15], x[03], x[07], x[11]] = Self::qr(x[15], x[03], x[07], x[11]);

            [x[00], x[01], x[02], x[03]] = Self::qr(x[00], x[01], x[02], x[03]);
            [x[05], x[06], x[07], x[04]] = Self::qr(x[05], x[06], x[07], x[04]);
            [x[10], x[11], x[08], x[09]] = Self::qr(x[10], x[11], x[08], x[09]);
            [x[15], x[12], x[13], x[14]] = Self::qr(x[15], x[12], x[13], x[14]);
        }
        self.x = x;
        x.map(|x| x.value())
    }
}

pub type Salsa8 = Salsa<8>;
impl_ring_rng32!(Salsa8, 16, next_raw);

pub type Salsa20 = Salsa<20>;
impl_ring_rng32!(Salsa20, 16, next_raw);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safe_test;

    safe_test!(Salsa8);
    safe_test!(Salsa20);
}
