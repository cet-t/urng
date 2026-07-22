use wrapn::{Wrap, wrap};

use crate::_internal::impl_seed;
#[allow(unused_imports)]
use crate::{_internal::FSCALE32, rng::Rng32, rng32::SplitMix32};

// --- Threefry32x4 ---

const THREEFRY32_C240: u32 = 0x1BD11BDA;
/// A Threefry4x32 random number generator (Random123 family).
///
/// This is a counter-based RNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 4 output values per block. Implements [`Rng32`] directly:
/// each call to [`Rng32::nextu`] hands out one `u32` from an internal 4-word
/// buffer, recomputing a fresh block every 4th call.
///
/// # Examples
///
/// ```
/// use urng::rng32::Threefry32x4;
/// use urng::rng::Rng32;
///
/// let mut rng = Threefry32x4::new(1);
/// let _: u32 = rng.nextu();
/// ```
#[repr(C, align(64))]
pub struct Threefry32x4 {
    pub(crate) c: [Wrap<u32>; 4],
    pub(crate) k: [Wrap<u32>; 5],
    pub(crate) tw: [Wrap<u32>; 3],
    pub(crate) pos: Wrap<usize>,
    pub(crate) buf: [Wrap<u32>; 4],
}

impl Threefry32x4 {
    /// Creates a new `Threefry32x4` instance seeded with the given value.
    ///
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32::new(seed);
        let mut k = wrap![0u32; 5];
        for item in k.iter_mut().take(4) {
            *item = seedgen.nextu().into();
        }
        k[4] = k[0] ^ k[1] ^ k[2] ^ k[3] ^ THREEFRY32_C240;

        let tw0 = seedgen.nextu();
        let tw1 = seedgen.nextu();
        let tw = wrap![tw0, tw1, tw0 ^ tw1];

        Self {
            c: wrap![0; 4],
            k,
            tw,
            pos: 4.into(),
            buf: wrap![0; 4],
        }
    }

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 4 output words.
    ///
    /// # Arguments
    ///
    /// * `c`  - 4-word counter (the plaintext block).
    /// * `k`  - 5-word key schedule (k\[4\] = parity word).
    /// * `tw` - 3-word tweak schedule (tw\[2\] = tw\[0\] ^ tw\[1\]).
    #[inline(always)]
    pub(crate) fn compute(c: [u32; 4], k: &[u32; 5], tw: &[u32; 3]) -> [u32; 4] {
        let mut v = c.map(|x| wrap!(x));

        macro_rules! round {
            ($r_sh_0:expr, $r_sh_1:expr) => {
                let y0 = v[0] + v[1];
                let f1 = v[1].rotate_left($r_sh_0) ^ y0;

                let y1 = v[2] + v[3];
                let f3 = v[3].rotate_left($r_sh_1) ^ y1;

                v[0] = y0;
                v[1] = f3;
                v[2] = y1;
                v[3] = f1;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] += k[$s % 5];
                v[1] += wrap!(k[($s + 1) % 5]) + tw[$s % 3];
                v[2] += wrap!(k[($s + 2) % 5]) + tw[($s + 1) % 3];
                v[3] += wrap!(k[($s + 3) % 5]) + $s as u32;
            };
        }

        inject_key!(0);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(1);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(2);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        inject_key!(3);
        round!(6, 20);
        round!(17, 11);
        round!(25, 10);
        round!(18, 20);

        inject_key!(4);
        round!(10, 26);
        round!(11, 21);
        round!(13, 27);
        round!(23, 5);

        let ksi5_0 = k[0];
        let ksi5_1 = k[1].wrapping_add(tw[2]);
        let ksi5_2 = k[2].wrapping_add(tw[0]);
        let ksi5_3 = k[3].wrapping_add(5);

        [
            (v[0] + ksi5_0) ^ c[0],
            (v[1] + ksi5_1) ^ c[1],
            (v[2] + ksi5_2) ^ c[2],
            (v[3] + ksi5_3) ^ c[3],
        ]
        .map(|x| x.value())
    }

    /// Generates the next block of 4 random `u32` values in one call.
    ///
    /// This is the raw bulk-generation path (used internally to refill the
    /// scalar [`Rng32::nextu`] buffer, and available directly for
    /// throughput-sensitive callers that want the whole block at once).
    #[inline(always)]
    pub fn next_raw(&mut self) -> [u32; 4] {
        let dst = Self::compute(
            self.c.map(|x| x.value()),
            &self.k.map(|x| x.value()),
            &self.tw.map(|x| x.value()),
        );

        self.c[0] += 1;
        if self.c[0] == 0 {
            self.c[1] += 1;
            if self.c[1] == 0 {
                self.c[2] += 1;
                if self.c[2] == 0 {
                    self.c[3] += 1;
                }
            }
        }

        dst
    }
}

impl_seed!(Threefry32x4, 32);

crate::_internal::impl_ring_rng32!(Threefry32x4, 4, next_raw);

// --- Threefry32x2 ---

/// A Threefry 2x32 random number generator (Random123 family).
///
/// Counter-based PRNG using a reduced-round (20 rounds) Threefish cipher
/// with 32-bit words and 2 output values per block.
///
/// # Examples
///
/// ```
/// use urng::rng32::Threefry32x2;
/// use urng::rng::Rng32;
///
/// let mut rng = Threefry32x2::new(1);
/// let _: u32 = rng.nextu();
/// ```
pub struct Threefry32x2 {
    pub(crate) c: [Wrap<u32>; 2],
    pub(crate) k: [Wrap<u32>; 3],
    pub(crate) buf: [Wrap<u32>; 2],
    pub(crate) pos: Wrap<usize>,
}

impl Threefry32x2 {
    /// Creates a new `Threefry32x2` instance seeded with the given value.
    ///
    #[inline]
    pub fn new(seed: u32) -> Self {
        let mut sm = SplitMix32::new(seed);
        let k0 = sm.nextu();
        let k1 = sm.nextu();

        Self {
            c: wrap![0, 0],
            k: wrap![k0, k1, k0 ^ k1 ^ THREEFRY32_C240],
            buf: wrap![0; 2],
            pos: 2.into(),
        }
    }

    /// Pure counter-based computation: applies 20 rounds of Threefish to produce 2 output words.
    ///
    /// # Arguments
    ///
    /// * `c` - 2-word counter (the plaintext block).
    /// * `k` - 3-word key schedule (k[2] = k[0] ^ k[1] ^ C240).
    #[inline(always)]
    pub(crate) fn compute(c: [u32; 2], k: &[u32; 3]) -> [u32; 2] {
        let mut v = c.map(|x| wrap!(x));

        macro_rules! round {
            ($r_sh:expr) => {
                let y = v[0] + v[1];
                v[0] = y;
                v[1] = v[1].rotate_left($r_sh) ^ y;
            };
        }

        macro_rules! inject_key {
            ($s:expr) => {
                v[0] += k[$s % 3];
                v[1] = v[1] + k[($s + 1) % 3] + $s as u32;
            };
        }

        inject_key!(0);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(1);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(2);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        inject_key!(3);
        round!(17);
        round!(29);
        round!(16);
        round!(24);

        inject_key!(4);
        round!(13);
        round!(15);
        round!(26);
        round!(6);

        let ksi5_0 = k[2];
        let ksi5_1 = k[0].wrapping_add(5);

        [v[0] + ksi5_0, v[1] + ksi5_1].map(|x| x.value())
    }

    /// Generates the next block of 2 random `u32` values in one call.
    ///
    /// This is the raw bulk-generation path (used internally to refill the
    /// scalar [`Rng32::nextu`] buffer, and available directly for
    /// throughput-sensitive callers that want the whole block at once).
    #[inline(always)]
    pub fn next_raw(&mut self) -> [u32; 2] {
        let k = self.k.map(|x| x.value());
        let dst = Self::compute(self.c.map(|x| x.value()), &k);
        self.k
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = k[i].into());
        let (n_c0, overflow) = self.c[0].value().overflowing_add(1);
        self.c[0] = n_c0.into();
        if overflow {
            self.c[1] += 1;
        }
        dst
    }
}

crate::_internal::impl_ring_rng32!(Threefry32x2, 2, next_raw);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Threefry32x4);
    crate::safe_test!(Threefry32x2);
}
