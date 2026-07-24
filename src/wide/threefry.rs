use crate::wide::impl_methods;
use crate::{Rng, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

const THREEFRY32_C240: u32 = 0x1BD11BDA;

macro_rules! impl_threefry32x2_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("Threefry 2x32 producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Threefry32x2`]. A counter-based generator"]
            #[doc = "using a reduced-round (20-round) Threefish cipher with 2 output words per block; each"]
            #[doc = "`nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Threefry32x2x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Threefry32x2x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Threefry32x2x $size>] {
                c0: [<u32x $size>],
                c1: [<u32x $size>],
                k0: [<u32x $size>],
                k1: [<u32x $size>],
                k2: [<u32x $size>],
                buf0: [<u32x $size>],
                buf1: [<u32x $size>],
                pos: usize,
            }

            #[allow(dead_code)]
            impl [<Threefry32x2x $size>] {
                #[doc = "Creates a new generator, deriving the 2-word key schedule and counter from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    let k0 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let k1 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    Self {
                        c0: [<u32x $size>]::splat(0),
                        c1: [<u32x $size>]::splat(0),
                        k0,
                        k1,
                        k2: k0 ^ k1 ^ [<u32x $size>]::splat(THREEFRY32_C240),
                        buf0: [<u32x $size>]::splat(0),
                        buf1: [<u32x $size>]::splat(0),
                        pos: 2,
                    }
                }

                #[doc = "Pure 20-round Threefish encryption of the current counter, returning the two output words."]
                #[inline(always)]
                fn compute(&self) -> ([<u32x $size>], [<u32x $size>]) {
                    let mut v0 = self.c0;
                    let mut v1 = self.c1;

                    macro_rules! round {
                        ($r:expr) => {{
                            let y = v0 + v1;
                            v0 = y;
                            v1 = ((v1 << $r) | (v1 >> (32 - $r))) ^ y;
                        }};
                    }
                    macro_rules! inject_key {
                        ($s:expr, $a:expr, $b:expr) => {{
                            v0 += $a;
                            v1 += $b + [<u32x $size>]::splat($s as u32);
                        }};
                    }

                    inject_key!(0, self.k0, self.k1);
                    round!(13); round!(15); round!(26); round!(6);
                    inject_key!(1, self.k1, self.k2);
                    round!(17); round!(29); round!(16); round!(24);
                    inject_key!(2, self.k2, self.k0);
                    round!(13); round!(15); round!(26); round!(6);
                    inject_key!(3, self.k0, self.k1);
                    round!(17); round!(29); round!(16); round!(24);
                    inject_key!(4, self.k1, self.k2);
                    round!(13); round!(15); round!(26); round!(6);

                    (
                        v0 + self.k2,
                        v1 + self.k0 + [<u32x $size>]::splat(5),
                    )
                }

                #[doc = "Recomputes a fresh block of 2 values and advances the counter when the buffer is exhausted."]
                #[inline(always)]
                fn refill(&mut self) {
                    let (buf0, buf1) = self.compute();
                    self.buf0 = buf0;
                    self.buf1 = buf1;
                    self.c0 += [<u32x $size>]::splat(1);
                    self.pos = 0;
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane, refilling as needed."]
                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    if self.pos >= 2 {
                        self.refill();
                    }
                    let out = if self.pos == 0 { self.buf0 } else { self.buf1 };
                    self.pos += 1;
                    bytemuck::cast(out)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_threefry32x2_variants!($size);)+
    };
}

macro_rules! impl_threefry32x4_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("Threefry 4x32 producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Threefry32x4`]. A counter-based generator"]
            #[doc = "using a reduced-round (20-round) Threefish cipher with 4 output words per block; each"]
            #[doc = "`nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Threefry32x4x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Threefry32x4x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Threefry32x4x $size>] {
                c0: [<u32x $size>],
                c1: [<u32x $size>],
                c2: [<u32x $size>],
                c3: [<u32x $size>],
                k0: [<u32x $size>],
                k1: [<u32x $size>],
                k2: [<u32x $size>],
                k3: [<u32x $size>],
                k4: [<u32x $size>],
                tw0: [<u32x $size>],
                tw1: [<u32x $size>],
                tw2: [<u32x $size>],
                buf0: [<u32x $size>],
                buf1: [<u32x $size>],
                buf2: [<u32x $size>],
                buf3: [<u32x $size>],
                pos: usize,
            }

            #[allow(dead_code)]
            impl [<Threefry32x4x $size>] {
                #[doc = "Creates a new generator, deriving the 4-word key schedule, tweak and counter from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    let k0 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let k1 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let k2 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let k3 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let tw0 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    let tw1 = [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu()));
                    Self {
                        c0: [<u32x $size>]::splat(0),
                        c1: [<u32x $size>]::splat(0),
                        c2: [<u32x $size>]::splat(0),
                        c3: [<u32x $size>]::splat(0),
                        k0,
                        k1,
                        k2,
                        k3,
                        k4: k0 ^ k1 ^ k2 ^ k3 ^ [<u32x $size>]::splat(THREEFRY32_C240),
                        tw0,
                        tw1,
                        tw2: tw0 ^ tw1,
                        buf0: [<u32x $size>]::splat(0),
                        buf1: [<u32x $size>]::splat(0),
                        buf2: [<u32x $size>]::splat(0),
                        buf3: [<u32x $size>]::splat(0),
                        pos: 4,
                    }
                }

                #[doc = "Pure 20-round Threefish encryption of the current counter, returning the four output words."]
                #[inline(always)]
                fn compute(&self) -> ([<u32x $size>], [<u32x $size>], [<u32x $size>], [<u32x $size>]) {
                    let mut v0 = self.c0;
                    let mut v1 = self.c1;
                    let mut v2 = self.c2;
                    let mut v3 = self.c3;

                    macro_rules! round {
                        ($r0:expr, $r1:expr) => {{
                            let y0 = v0 + v1;
                            let f1 = ((v1 << $r0) | (v1 >> (32 - $r0))) ^ y0;
                            let y1 = v2 + v3;
                            let f3 = ((v3 << $r1) | (v3 >> (32 - $r1))) ^ y1;
                            v0 = y0;
                            v1 = f3;
                            v2 = y1;
                            v3 = f1;
                        }};
                    }
                    macro_rules! inject_key {
                        ($s:expr, $k0:expr, $k1:expr, $k2:expr, $k3:expr, $tw0:expr, $tw1:expr) => {{
                            v0 += $k0;
                            v1 += $k1 + $tw0;
                            v2 += $k2 + $tw1;
                            v3 += $k3 + [<u32x $size>]::splat($s as u32);
                        }};
                    }

                    inject_key!(0, self.k0, self.k1, self.k2, self.k3, self.tw0, self.tw1);
                    round!(10, 26); round!(11, 21); round!(13, 27); round!(23, 5);
                    inject_key!(1, self.k1, self.k2, self.k3, self.k4, self.tw1, self.tw2);
                    round!(6, 20); round!(17, 11); round!(25, 10); round!(18, 20);
                    inject_key!(2, self.k2, self.k3, self.k4, self.k0, self.tw2, self.tw0);
                    round!(10, 26); round!(11, 21); round!(13, 27); round!(23, 5);
                    inject_key!(3, self.k3, self.k4, self.k0, self.k1, self.tw0, self.tw1);
                    round!(6, 20); round!(17, 11); round!(25, 10); round!(18, 20);
                    inject_key!(4, self.k4, self.k0, self.k1, self.k2, self.tw1, self.tw2);
                    round!(10, 26); round!(11, 21); round!(13, 27); round!(23, 5);

                    (
                        (v0 + self.k0) ^ self.c0,
                        (v1 + self.k1 + self.tw2) ^ self.c1,
                        (v2 + self.k2 + self.tw0) ^ self.c2,
                        (v3 + self.k3 + [<u32x $size>]::splat(5)) ^ self.c3,
                    )
                }

                #[doc = "Recomputes a fresh block of 4 values and advances the counter when the buffer is exhausted."]
                #[inline(always)]
                fn refill(&mut self) {
                    let (buf0, buf1, buf2, buf3) = self.compute();
                    self.buf0 = buf0;
                    self.buf1 = buf1;
                    self.buf2 = buf2;
                    self.buf3 = buf3;
                    self.c0 += [<u32x $size>]::splat(1);
                    self.pos = 0;
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane, refilling as needed."]
                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    if self.pos >= 4 {
                        self.refill();
                    }
                    let out = match self.pos {
                        0 => self.buf0,
                        1 => self.buf1,
                        2 => self.buf2,
                        _ => self.buf3,
                    };
                    self.pos += 1;
                    bytemuck::cast(out)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_threefry32x4_variants!($size);)+
    };
}

impl_threefry32x2_variants!(4, 8, 16);
impl_threefry32x4_variants!(4, 8, 16);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Threefry32x2x4);
    crate::safe_test!(Threefry32x2x8);
    crate::safe_test!(Threefry32x2x16);
    crate::safe_test!(Threefry32x4x4);
    crate::safe_test!(Threefry32x4x8);
    crate::safe_test!(Threefry32x4x16);
}
