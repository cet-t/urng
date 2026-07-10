use ::wide::{u32x4, u32x8, u32x16};

use crate::wide::impl_methods;

macro_rules! impl_variants {
    ($size:expr) => {
        paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<SplitMix32x $size>] {
                state: [<u32x $size>],
            }

            #[allow(dead_code)]
            impl [<SplitMix32x $size>] {
                pub fn new(seed: u32) -> Self {
                    let mut state = [0u32; $size];
                    state.iter_mut().fold(seed | 1, |s, x| {
                        *x = s;
                        s.wrapping_add(0x9E3779B9)
                    });

                    Self {
                        state: [<u32x $size>]::from(state),
                    }
                }

                #[inline(always)]
                pub(crate) fn compute(state: [<u32x $size>]) -> [<u32x $size>] {
                    let mut z = state;
                    z = (z ^ (z >> 16)) * [<u32x $size>]::splat(0x85EBCA6B);
                    z = (z ^ (z >> 13)) * [<u32x $size>]::splat(0xC2B2AE35);
                    z ^ (z >> 16)
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    self.state += [<u32x $size>]::splat(0x9E3779B9);
                    let z = Self::compute(self.state);
                    bytemuck::cast(z)
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

    crate::safe_test!(SplitMix32x4);
    crate::safe_test!(SplitMix32x8);
    crate::safe_test!(SplitMix32x16);
}
