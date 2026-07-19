use crate::wide::{impl_methods, wide_rotate_right};
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("SFC32 (Small Fast Counter) producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::rng32::Sfc32`]. Uses a 128-bit state plus an"]
            #[doc = "internal counter; each `nextu` call returns an array of `u32`, one per lane."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = concat!("use urng::wide::Sfc32x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Sfc32x", stringify!($size), "::new(1);")]
            #[doc = concat!("let v = rng.nextu();")]
            #[doc = concat!("assert_eq!(v.len(), ", stringify!($size), ");")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Sfc32x $size>] {
                a: [<u32x $size>],
                b: [<u32x $size>],
                c: [<u32x $size>],
                counter: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<Sfc32x $size>] {
                #[doc = "Creates a new generator, seeding `a`, `b`, `c` and `counter` of every lane from `seed`."]
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);

                    Self {
                        a: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        b: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        c: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        counter: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Pure SFC32 scramble applied to a single `tmp` word and the per-lane state references."]
                #[inline(always)]
                pub(crate) fn compute(
                    tmp: [<u32x $size>],
                    a: &mut [<u32x $size>],
                    b: &mut [<u32x $size>],
                    c: &mut [<u32x $size>],
                    counter: &mut [<u32x $size>],
                ) -> [<u32x $size>] {
                    *counter += [<u32x $size>]::splat(1);
                    *a = *b ^ (*b >> 9);
                    *b = *c + (*c << 3);
                    *c = wide_rotate_right!(32 *c, 11);
                    *c += tmp;
                    tmp
                }

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[doc = ""]
                #[doc = "Mixes the state, advances the counter, and returns the previous `a + b + counter` sum."]
                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let tmp = self.a + self.b + self.counter;
                    let result = Self::compute(
                        tmp,
                        &mut self.a,
                        &mut self.b,
                        &mut self.c,
                        &mut self.counter,
                    );
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

    crate::safe_test!(Sfc32x4);
    crate::safe_test!(Sfc32x8);
    crate::safe_test!(Sfc32x16);
}
