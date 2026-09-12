use ::wide::{u64x4, u64x8};

use crate::wide::{RngW, wide_rotate_left};
use crate::{Rng, SplitMix64};

macro_rules! impl_squares32_variants {
    ($size:expr) => {
        ::pastey::paste! {
            #[doc = concat!("Squares32 producing ", stringify!($size), " values per call via `wide` SIMD vectors.")]
            #[doc = ""]
            #[doc = "Portable-SIMD counterpart of [`crate::cbrng::b32::Squares32`]. A counter-based generator that"]
            #[doc = "runs four rounds of the middle-square scramble over wide `u64` state; each `nextu` call"]
            #[doc = "returns an array of `u32`."]
            #[doc = ""]
            #[doc = "# Example"]
            #[doc = "```"]
            #[doc = "use urng::wide::WRng;"]
            #[doc = concat!("use urng::wide::Squares32x", stringify!($size), ";")]
            #[doc = ""]
            #[doc = concat!("let mut rng = Squares32x", stringify!($size), "::new(0);")]
            #[doc = concat!("let _ = rng.nextu();")]
            #[doc = "```"]
            #[allow(dead_code)]
            #[repr(C, align(64))]
            pub struct [<Squares32x $size>] {
                c: [<u64x $size>],
                k: [<u64x $size>],
            }

            #[allow(dead_code)]
            impl [<Squares32x $size>] {
                #[doc = "Creates a new generator, seeding every lane's key from `seed` with per-lane counters."]
                pub fn new(seed: u64) -> Self {
                    Self::with_counter(seed, 0)
                }

                #[doc = "Builds a generator from `seed` with explicit starting counters (used to split lanes for `x16`)."]
                fn with_counter(seed: u64, counter: u64) -> Self {
                    let mut seedgen = SplitMix64::new(seed | 1);
                    Self {
                        c: [<u64x $size>]::from(std::array::from_fn(|i| counter + i as u64)),
                        k: [<u64x $size>]::from([0u64; $size].map(|_| seedgen.nextu())),
                    }
                }

                #[doc = "Four-round middle-square computation producing the high `u32` of the final mix."]
                #[inline(always)]
                fn compute_yz(y: [<u64x $size>], z: [<u64x $size>]) -> [u32; $size] {
                    let mut x = y * y + y;
                    x = wide_rotate_left!(64 x, 32);
                    x = x * x + z;
                    x = wide_rotate_left!(64 x, 32);
                    x = x * x + y;
                    x = wide_rotate_left!(64 x, 32);
                    let out: [u64; $size] = ((x * x + z) >> 32u64).to_array();
                    out.map(|x| x as u32)
                }

            }

            impl RngW<$size> for [<Squares32x $size>] {
                type Word = u32;

                #[doc = "Generates the next block of `u32` values, one per SIMD lane."]
                #[inline(always)]
                fn nextu(&mut self) -> [Self::Word; $size] {
                    let y = self.c * self.k;
                    let z = y + self.k;
                    self.c += [<u64x $size>]::splat($size as u64);
                    bytemuck::cast(Self::compute_yz(y, z))
                }
            }
        }
    };
    ($($size:expr),+) => {
        $(impl_squares32_variants!($size);)+
    };
}

impl_squares32_variants!(4, 8);

#[cfg(test)]
mod tests {
    use super::*;

    crate::safe_test! {
        Squares32x4,
        Squares32x8
    }
}
