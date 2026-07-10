use crate::wide::{impl_methods, wide_rotate_left};
use crate::{Rng64, SplitMix64};
use ::wide::{u64x4, u64x8};

macro_rules! impl_squares32_variants {
    ($size:expr, $lanes:expr) => {
        ::paste::paste! {
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Squares32x $size>] {
                c: [<u64x $lanes>],
                k: [<u64x $lanes>],
            }

            #[allow(dead_code)]
            impl [<Squares32x $size>] {
                pub fn new(seed: u64) -> Self {
                    Self::with_counter(seed, 0)
                }

                fn with_counter(seed: u64, counter: u64) -> Self {
                    let mut seedgen = SplitMix64::new(seed | 1);
                    Self {
                        c: [<u64x $lanes>]::from(std::array::from_fn(|i| counter + i as u64)),
                        k: [<u64x $lanes>]::from([0u64; $lanes].map(|_| seedgen.nextu())),
                    }
                }

                #[inline(always)]
                fn compute_yz(y: [<u64x $lanes>], z: [<u64x $lanes>]) -> [u32; $lanes] {
                    let mut x = y * y + y;
                    x = wide_rotate_left!(64 x, 32);
                    x = x * x + z;
                    x = wide_rotate_left!(64 x, 32);
                    x = x * x + y;
                    x = wide_rotate_left!(64 x, 32);
                    let out: [u64; $lanes] = ((x * x + z) >> 32u64).to_array();
                    out.map(|x| x as u32)
                }

                #[inline(always)]
                pub fn nextu(&mut self) -> [u32; $size] {
                    let y = self.c * self.k;
                    let z = y + self.k;
                    self.c += [<u64x $lanes>]::splat($lanes as u64);
                    bytemuck::cast(Self::compute_yz(y, z))
                }

                impl_methods!($size, 32);
            }
        }
    };
}

impl_squares32_variants!(4, 4);
impl_squares32_variants!(8, 8);

#[allow(dead_code)]
#[repr(C, align(64))]
pub struct Squares32x16 {
    lo: Squares32x8,
    hi: Squares32x8,
}

#[allow(dead_code)]
impl Squares32x16 {
    pub fn new(seed: u64) -> Self {
        Self {
            lo: Squares32x8::with_counter(seed, 0),
            hi: Squares32x8::with_counter(seed, 8),
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

    crate::safe_test!(Squares32x4, Squares32x4::new(0));
    crate::safe_test!(Squares32x8, Squares32x8::new(0));
    crate::safe_test!(Squares32x16, Squares32x16::new(0));
}
