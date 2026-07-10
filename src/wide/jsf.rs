use crate::wide::{SplitMix32x4, SplitMix32x8, SplitMix32x16, impl_methods, wide_rotate_left};
use ::wide::{u32x4, u32x8, u32x16};

macro_rules! impl_variants {
    ($size:expr) => {
        ::paste::paste! {
            #[repr(C, align(64))]
            pub struct [<Jsf32x $size>] {
                pub(crate) a: [<u32x $size>],
                pub(crate) b: [<u32x $size>],
                pub(crate) c: [<u32x $size>],
                pub(crate) d: [<u32x $size>],
            }

            impl [<Jsf32x $size>] {
                pub fn new(seed: u32) -> Self {
                    let mut seedgen = [<SplitMix32x $size>]::new(seed);
                    Self {
                        a: [<u32x $size>]::splat(0xf1ea5eed),
                        b: bytemuck::cast(seedgen.nextu()),
                        c: bytemuck::cast(seedgen.nextu()),
                        d: bytemuck::cast(seedgen.nextu()),
                    }
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let e = self.a - wide_rotate_left!(32 self.b, 27);
                    self.a = self.b ^ wide_rotate_left!(32 self.c, 17);
                    self.b = self.c + self.d;
                    self.c = self.d + e;
                    self.d = e + self.a;
                    bytemuck::cast(self.d)
                }

                impl_methods!($size, 32);
            }
        }
    };
    ($($size:expr),+ $(,)*) => {
        $(impl_variants!($size);)+
    }
}

impl_variants!(4, 8, 16);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Jsf32x4);
    crate::safe_test!(Jsf32x8);
    crate::safe_test!(Jsf32x16);
}
