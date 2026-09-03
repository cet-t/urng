use crate::rng::{Rng, Word};

/// Random element selection from a slice, for any [`Rng`].
pub trait Choice: Rng {
    /// Returns a random element from a slice.
    #[inline(always)]
    fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
        let index = self.nextu().to_index(choices.len());
        &choices[index]
    }

    /// Returns a random mutable element from a slice.
    #[inline(always)]
    fn choice_mut<'a, T>(&mut self, choices: &'a mut [T]) -> &'a mut T {
        let index = self.nextu().to_index(choices.len());
        &mut choices[index]
    }
}

impl<R: Rng + ?Sized> Choice for R {}

#[cfg(test)]
mod tests {
    use crate::Choice;

    #[test]
    fn it_works() {
        let mut rng = crate::Sfc32::new(0);
        let items: Vec<_> = (0..10).collect();
        assert_eq!(rng.choice(&items), &items[3]);
    }
}
