use crate::wide::impl_methods;
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_xorshift32_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("Xorshift32 producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Xorshift32`]. A shift-register generator;"]
            #[doc = "each `nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Xorshift32x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Xorshift32x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Xorshift32x $size>] {
                a: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Xorshift32x $size>] {
                #[doc = "Creates a new generator, seeding the single shift-register word of every lane from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        a: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Applies the Xorshift32 scramble: `x ^= x << 13; x ^= x >> 17; x ^= x << 5`."]
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
        ::pastey::paste! {
            #[doc = concat!("Xorshift128 producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Xorshift128`]. A 128-bit internal state;"]
            #[doc = "each `nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Xorshift128x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Xorshift128x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
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
                #[doc = "Creates a new generator, seeding the four state words of every lane from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        x0: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x1: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x2: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        x3: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Applies the Xorshift128 scramble over the 128-bit state."]
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
        ::pastey::paste! {
            #[doc = concat!("Xorwow producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Xorwow`]. Combines a Xorshift state with a"]
            #[doc = "Weyl (linear) counter; each `nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Xorwowx", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Xorwowx", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
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
                #[doc = "Creates a new generator, seeding the five state words and the Weyl counter of every lane from `seed`."]
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

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Applies the Xorwow scramble and adds the advancing Weyl counter."]
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
