use crate::wide::{impl_methods, wide_rotate_left};
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("Xoroshiro64** producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Xoroshiro64Ss`]. A 64-bit state (two"]
            #[doc = "32-bit words) per lane; each `nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Xoroshiro64Ssx", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Xoroshiro64Ssx", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Xoroshiro64Ssx $size>] {
                s0: [<u32x $size>],
                s1: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Xoroshiro64Ssx $size>] {
                #[doc = "Creates a new generator, seeding the two state words of every lane from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        s0: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        s1: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Applies the Xoroshiro64** scramble: `rotl(s0 * 0x9E3779BB, 5) * 5`."]
                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let s0 = self.s0;
                    let mut s1 = self.s1;
                    let result = wide_rotate_left!(32 (s0 * 0x9E3779BBu32), 5) * 5u32;

                    s1 ^= s0;
                    self.s0 = wide_rotate_left!(32 s0, 26) ^ s1 ^ (s1 << 9);
                    self.s1 = wide_rotate_left!(32 s1, 13);

                    bytemuck::cast(result)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_variants!($size);)+
    };
}

impl_variants!(4, 8, 16);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xoroshiro64Ssx4);
    crate::safe_test!(Xoroshiro64Ssx8);
    crate::safe_test!(Xoroshiro64Ssx16);
}
