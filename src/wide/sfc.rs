use crate::{_internal::FSCALE32, wide::SplitMix32x4};
use ::wide::u32x8;

#[repr(C, align(64))]
pub struct Sfc32x8 {
    a: u32x8,
    b: u32x8,
    c: u32x8,
    counter: u32x8,
}

impl Sfc32x8 {
    #[target_feature(enable = "avx2")]
    pub fn new(seed: u32) -> Self {
        let mut seedgen = SplitMix32x4::new(seed);
        let mut a = [0u32; 8];
        let mut b = [0u32; 8];
        let mut c = [0u32; 8];
        let mut counter = [0u32; 8];
        for i in 0..a.len() {
            [a[i], b[i], c[i], counter[i]] = seedgen.nextu();
        }

        Self {
            a: u32x8::from(a),
            b: u32x8::from(b),
            c: u32x8::from(c),
            counter: u32x8::from(counter),
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn compute(
        tmp: u32x8,
        a: &mut u32x8,
        b: &mut u32x8,
        c: &mut u32x8,
        counter: &mut u32x8,
    ) -> u32x8 {
        *counter += u32x8::splat(1);
        *a = *b ^ (*b >> 9);
        *b = *c + (*c << 3);
        *c = (*c << 21) | (*c >> 11);
        *c += tmp;
        tmp
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn nextu(&mut self) -> [u32; 8] {
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

    #[target_feature(enable = "avx2")]
    #[inline]
    pub fn nextf(&mut self) -> [f32; 8] {
        self.nextu().map(|x| (x as f32) * FSCALE32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unsafe_test;

    unsafe_test!(Sfc32x8);
}
