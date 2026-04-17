//! Wide SIMD-accelerated random number generators.

use ::wide::{u32x4, u32x8};

macro_rules! sm32xn {
    ($v:expr, $intrin:expr) => {
        paste::paste! {
            #[repr(C, align(64))]
            pub struct [<SplitMix32 x $v>] {
                state: [<u32x $v>],
            }

            impl [<SplitMix32 x $v>] {
                #[target_feature(enable = $intrin)]
                pub fn new(seed: u32) -> Self {
                    let mut state = [0u32; $v];
                    state.iter_mut().fold(seed | 1, |s, x| {
                        *x = s;
                        s.wrapping_add(0x9E3779B9)
                    });

                    Self {
                        state: [<u32x $v>]::from(state),
                    }
                }

                #[target_feature(enable = $intrin)]
                #[inline]
                pub(crate) fn compute(state: [<u32x $v>]) -> [<u32x $v>] {
                    let mut z = state;
                    z = (z ^ (z >> 16)) * [<u32x $v>]::splat(0x85EBCA6B);
                    z = (z ^ (z >> 13)) * [<u32x $v>]::splat(0xC2B2AE35);
                    z ^ (z >> 16)
                }

                #[target_feature(enable = $intrin)]
                #[inline]
                pub fn nextu(&mut self) -> [u32; $v] {
                    self.state += [<u32x $v>]::splat(0x9E3779B9);
                    let z = Self::compute(self.state);
                    bytemuck::cast(z)
                }

                #[target_feature(enable = $intrin)]
                #[inline]
                pub fn nextf(&mut self) -> [f32; $v] {
                    self.nextu()
                        .map(|x| f32::from_bits((x >> 9) | 0x3F800000) - 1.0)
                }
            }
        }
    };
}

sm32xn!(4, "avx2");
sm32xn!(8, "avx512f");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unsafe_test;

    unsafe_test!(SplitMix32x4);
    unsafe_test!(SplitMix32x8);
}
