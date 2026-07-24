use crate::{rng::Rng, rng32::*};

crate::impl_rand_trait!(
    Mt19937,
    Sfmt607,
    Sfmt1279,
    Sfmt2281,
    Sfmt4253,
    Sfmt11213,
    Sfmt19937,
    Sfmt44497,
    Sfmt86243,
    Sfmt132049,
    Sfmt216091,
    Pcg32,
    Philox32x4,
    Sfc32,
    SplitMix32,
    Squares32,
    Xoroshiro64Ss,
    Xorshift32,
    Xorshift128,
    Xorwow,
    Xoshiro128Pp,
    Xoshiro128Ss
);

crate::impl_try_rng_trait!(
    Mt19937,
    Sfmt607,
    Sfmt1279,
    Sfmt2281,
    Sfmt4253,
    Sfmt11213,
    Sfmt19937,
    Sfmt44497,
    Sfmt86243,
    Sfmt132049,
    Sfmt216091,
    Pcg32,
    Sfc32,
    SplitMix32,
    Squares32,
    Xoroshiro64Ss,
    Xorshift32,
    Xorshift128,
    Xorwow,
    Xoshiro128Pp,
    Xoshiro128Ss
);

impl rand_core::TryRng for Philox32x4 {
    type Error = std::convert::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.nextu())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let out = self.next_raw();
        Ok((out[0] as u64) << 32 | out[1] as u64)
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let mut buf = [0u8; 16];
        let mut buf_pos = 16;
        let mut i = 0;
        while i < dst.len() {
            if buf_pos >= 16 {
                let arr = self.next_raw();
                for j in 0..4 {
                    buf[j * 4..(j + 1) * 4].copy_from_slice(&arr[j].to_le_bytes());
                }
                buf_pos = 0;
            }
            let take = (dst.len() - i).min(16 - buf_pos);
            dst[i..i + take].copy_from_slice(&buf[buf_pos..buf_pos + take]);
            i += take;
            buf_pos += take;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn sfmt19937_works() {
        let mut rng0 = Sfmt19937::seed_from_u64(0);
        let mut rng1 = Sfmt19937::seed_from_u64(0);
        assert_eq!(rng0.next_u32(), rng1.next_u32());
    }

    #[test]
    fn philox32x4_works() {
        let mut rng0 = Philox32x4::seed_from_u64(0);
        let mut rng1 = Philox32x4::seed_from_u64(0);
        assert_eq!(rng0.next_u32(), rng1.next_u32());
    }
}
