use crate::wide::impl_methods;
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_xorshift32_variants {
    ($size:expr) => {
        ::paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Xorshift32x $size>] {
                a: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Xorshift32x $size>] {
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        a: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let x = self.a;
                    self.a = x ^ (x << 13);
                    self.a ^= self.a >> 17;
                    self.a ^= self.a << 5;
                    bytemuck::cast(self.a)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_xorshift32_variants!($size);)+
    };
}

macro_rules! impl_xorshift128_variants {
    ($size:expr) => {
        ::paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Xorshift128x $size>] {
                x0: [<u32x $size>],
                x1: [<u32x $size>],
                x2: [<u32x $size>],
                x3: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Xorshift128x $size>] {
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        x0: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x1: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x2: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x3: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let mut t = self.x3;
                    t ^= t << 11;
                    t ^= t >> 8;
                    let s = self.x0;
                    self.x3 = self.x2;
                    self.x2 = self.x1;
                    self.x1 = s;
                    self.x0 = t ^ s ^ (s >> 19);
                    bytemuck::cast(self.x0)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_xorshift128_variants!($size);)+
    };
}

macro_rules! impl_xorwow_variants {
    ($size:expr) => {
        ::paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Xorwowx $size>] {
                x0: [<u32x $size>],
                x1: [<u32x $size>],
                x2: [<u32x $size>],
                x3: [<u32x $size>],
                x4: [<u32x $size>],
                c: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Xorwowx $size>] {
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        x0: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x1: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x2: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x3: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x4: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        c: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let mut t = self.x4;
                    let s = self.x0;
                    self.x4 = self.x3;
                    self.x3 = self.x2;
                    self.x2 = self.x1;
                    self.x1 = s;

                    t ^= t >> 2;
                    t ^= t << 1;
                    t ^= s ^ (s << 4);
                    self.x0 = t;
                    self.c += [<u32x $size>]::splat(362437);
                    bytemuck::cast(t + self.c)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_xorwow_variants!($size);)+
    };
}

impl_xorshift32_variants!(4, 8, 16);
impl_xorshift128_variants!(4, 8, 16);
impl_xorwow_variants!(4, 8, 16);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xorshift32x4);
    crate::safe_test!(Xorshift32x8);
    crate::safe_test!(Xorshift32x16);
    crate::safe_test!(Xorshift128x4);
    crate::safe_test!(Xorshift128x8);
    crate::safe_test!(Xorshift128x16);
    crate::safe_test!(Xorwowx4);
    crate::safe_test!(Xorwowx8);
    crate::safe_test!(Xorwowx16);
}
