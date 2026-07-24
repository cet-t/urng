use crate::rng::{Rng, Word};

/// Random element selection from a slice, for any [`Rng`].
pub trait Choice: Rng {
    /// Returns a random element from a slice.
    #[inline(always)]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.nextu().to_index(choices.len());
        &choices[index]
    }

    #[inline(always)]
    fn choice_mut<'a, T>(&mut self, choices: &'a mut [T]) -> &'a mut T {
        let index = self.nextu().to_index(choices.len());
        &mut choices[index]
    }
}

impl<R: Rng + ?Sized> Choice for R {}
