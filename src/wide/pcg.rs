use crate::wide::impl_methods;
use crate::{Rng64, SplitMix64};
use ::wide::{u64x4, u64x8};

macro_rules! impl_pcg32_variants {
    ($size:expr, $lanes:expr) => {
        ::paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Pcg32x $size>] {
                state: [<u64x $lanes>],
                inc: [<u64x $lanes>],
            }

            #[allow(dead_code)]
            impl [<Pcg32x $size>] {
                pub fn new(seed: u64) -> Self {
                    let mut seedgen = SplitMix64::new(seed | 1);
                    Self {
                        state: [<u64x $lanes>]::from([0u64; $lanes].map(|_| seedgen.nextu())),
                        inc: [<u64x $lanes>]::from([0u64; $lanes].map(|_| seedgen.nextu())),
                    }
                }

                #[inline(always)]
                fn step(state: &mut [<u64x $lanes>], inc: [<u64x $lanes>]) -> [u32; $lanes] {
                    let oldstate = *state;
                    *state = oldstate * 6364136223846793005u64 + inc;
                    let xorshifted: [u64; $lanes] =
                        (((oldstate >> 18u64) ^ oldstate) >> 27u64).to_array();
                    let rot: [u64; $lanes] = (oldstate >> 59u64).to_array();
                    std::array::from_fn(|i| (xorshifted[i] as u32).rotate_right(rot[i] as u32))
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    bytemuck::cast(Self::step(&mut self.state, self.inc))
                }

                impl_methods!($size, 32);
            }
        }
    };
}

impl_pcg32_variants!(4, 4);
impl_pcg32_variants!(8, 8);

#[allow(dead_code)]
#[repr(C, align(64))]
pub struct Pcg32x16 {
    lo: Pcg32x8,
    hi: Pcg32x8,
}

#[allow(dead_code)]
impl Pcg32x16 {
    pub fn new(seed: u64) -> Self {
        Self {
            lo: Pcg32x8::new(seed),
            hi: Pcg32x8::new(SplitMix64::compute(seed ^ 0x9E3779B97F4A7C15)),
        }
    }

    #[inline(always)]
    pub fn nextu(&mut self) -> [u32; 16] {
        let lo = self.lo.nextu();
        let hi = self.hi.nextu();
        std::array::from_fn(|i| if i < 8 { lo[i] } else { hi[i - 8] })
    }

    impl_methods!(16, 32);
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test!(Pcg32x4, Pcg32x4::new(0));
    crate::safe_test!(Pcg32x8, Pcg32x8::new(0));
    crate::safe_test!(Pcg32x16, Pcg32x16::new(0));
}
