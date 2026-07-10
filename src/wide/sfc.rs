use crate::wide::{impl_methods, wide_rotate_right};
use crate::{Rng32, SplitMix32};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_variants {
    ($size:expr) => {
        ::paste::paste! {
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
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = SplitMix32::new(seed);

                    Self {
                        a: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        b: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        c: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                        counter: [<u32x $size>]::from([0u32; $size].map(|_| seedgen.nextu())),
                    }
                }

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
