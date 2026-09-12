use crate::{Rng, cbrng::b32::*, prng::b32::*};

crate::impl_rand_trait! {
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
    Philox32,
    Sfc32,
    SplitMix32,
    Squares32,
    Xoroshiro64Ss,
    Xorshift32,
    Xorshift128,
    Xorwow,
    Xoshiro128Pp,
    Xoshiro128Ss,
}

crate::impl_try_rng_trait! {
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
    Xoshiro128Ss,
    Philox32
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
    fn philox32_works() {
        let mut rng0 = Philox32::seed_from_u64(0);
        let mut rng1 = Philox32::seed_from_u64(0);
        assert_eq!(rng0.next_u32(), rng1.next_u32());
    }
}
