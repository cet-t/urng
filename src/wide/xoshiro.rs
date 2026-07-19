use crate::wide::{impl_methods, wide_rotate_left};
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_variants {
    ($name:ident, $scrambler:tt, $size:expr) => {
        ::pastey::paste! {
            #[doc = concat!(stringify!($name), " (xoshiro128) producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of the scalar xoshiro128 generators in `crate::rng32`."]
            #[doc = "A 128-bit state (four 32-bit words) per lane; each `nextu` call returns an array of"]
            #[doc = "`u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::", stringify!($name), "x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = ", stringify!($name), "x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<$name x $size>] {
                s0: [<u32x $size>],
                s1: [<u32x $size>],
                s2: [<u32x $size>],
                s3: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<$name x $size>] {
                #[doc = "Creates a new generator, seeding the four state words of every lane from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);
                    Self {
                        s0: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        s1: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        s2: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        s3: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Applies the xoshiro128 state update and the selected (`++` or `**`) scrambler."]
                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let res = impl_variants!(@scramble $scrambler, self.s0, self.s1, self.s3);
                    let t = self.s1 << 9;

                    self.s2 ^= self.s0;
                    self.s3 ^= self.s1;
                    self.s1 ^= self.s2;
                    self.s0 ^= self.s3;
                    self.s2 ^= t;
                    self.s3 = wide_rotate_left!(32 self.s3, 11);

                    bytemuck::cast(res)
                }

                impl_methods!($size, 32);
            }
        }
    };
    (@scramble pp, $s0:expr, $s1:expr, $s3:expr) => {
        wide_rotate_left!(32 ($s0 + $s3), 7) + $s0
    };
    (@scramble ss, $s0:expr, $s1:expr, $s3:expr) => {
        wide_rotate_left!(32 ($s1 * 5u32), 7) * 9u32
    };
    ($name:ident, $scrambler:tt; $($size:expr),+ $(,)*) => {
        $(impl_variants!($name, $scrambler, $size);)+
    };
}

impl_variants!(Xoshiro128Pp, pp; 4, 8, 16);
impl_variants!(Xoshiro128Ss, ss; 4, 8, 16);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Xoshiro128Ppx4);
    crate::safe_test!(Xoshiro128Ppx8);
    crate::safe_test!(Xoshiro128Ppx16);
    crate::safe_test!(Xoshiro128Ssx4);
    crate::safe_test!(Xoshiro128Ssx8);
    crate::safe_test!(Xoshiro128Ssx16);
}
