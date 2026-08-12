//! A wide (portable SIMD) random number generator trait.

use crate::rng::Word;

/// A wide random number generator producing `WIDTH` [`Word`]s per call.
///
/// Mirrors [`crate::rng::Rng`]: implementors provide only [`nextu`](WRng::nextu),
/// and the ranged/float draws are supplied as defaults derived from the
/// per-lane [`Word`] conversions, so the numeric formulas stay defined in
/// exactly one place. `WIDTH` is a named const generic (bind it as
/// `WRng<WIDTH = 8>`), so it reads at call sites the same way an associated
/// item would.
pub trait WRng<const WIDTH: usize> {
    /// The unsigned output word this generator produces, per lane.
    type Word: Word;

    /// Generates the next block of `WIDTH` raw words, each in `[0, 2^BITS)`.
    #[must_use]
    fn nextu(&mut self) -> [Self::Word; WIDTH];

    /// Generates `WIDTH` floats in `[0, 1)`.
    #[must_use]
    #[inline(always)]
    fn nextf(&mut self) -> [<Self::Word as Word>::Float; WIDTH] {
        self.nextu().map(Word::to_f01)
    }

    /// Generates `WIDTH` integers uniformly in the inclusive range `[min, max]`.
    #[must_use]
    #[inline(always)]
    fn randi(
        &mut self,
        min: <Self::Word as Word>::Int,
        max: <Self::Word as Word>::Int,
    ) -> [<Self::Word as Word>::Int; WIDTH] {
        self.nextu().map(|w| w.to_randi(min, max))
    }

    /// Generates `WIDTH` floats uniformly in the half-open range `[min, max)`.
    #[must_use]
    #[inline(always)]
    fn randf(
        &mut self,
        min: <Self::Word as Word>::Float,
        max: <Self::Word as Word>::Float,
    ) -> [<Self::Word as Word>::Float; WIDTH] {
        self.nextu().map(|w| w.to_randf(min, max))
    }
}

#[cfg(test)]
mod tests {
    use crate::Seed;
    use crate::wide::{Jsf32x4, Pcg32x4, WRng};

    #[test]
    fn seed_and_default_are_wired() {
        let mut a = Jsf32x4::from_seed(1);
        let mut b = Jsf32x4::from_seed(1);
        assert_eq!(a.nextu(), b.nextu());

        let _: Jsf32x4 = Default::default();

        let mut c = Pcg32x4::from_seed(1);
        let mut d = Pcg32x4::from_seed(1);
        assert_eq!(c.nextu(), d.nextu());

        let _: Pcg32x4 = Default::default();
    }
}
