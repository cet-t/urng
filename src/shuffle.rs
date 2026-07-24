#[derive(Debug)]
pub struct SliceShuffleError(pub(crate) ());
impl std::fmt::Display for SliceShuffleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot shuffle an empty slice.")
    }
}
impl std::error::Error for SliceShuffleError {}

pub type ShuffleResult<T> = std::result::Result<T, self::SliceShuffleError>;

/// An iterator that yields references to the elements of a slice in a shuffled order.
///
/// Returned by [`Shuffle::shuffled`].
pub struct ShuffledIter<'a, T> {
    pub(crate) slice: &'a [T],
    pub(crate) indices: Vec<usize>,
    pub(crate) pos: usize,
}

impl<'a, T> Iterator for ShuffledIter<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a T> {
        let index = *self.indices.get(self.pos)?;
        self.pos += 1;
        Some(&self.slice[index])
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len() - self.pos;
        (remaining, Some(remaining))
    }
}

use crate::rng::{Rng, Word};

/// Shuffles a slice in-place or returns a shuffled iterator, for any [`Rng`].
pub trait Shuffle: Rng {
    /// Shuffles `src` in place.
    #[inline(always)]
    fn shuffle<T>(&mut self, src: &mut [T]) -> crate::ShuffleResult<()> {
        if src.is_empty() {
            Err(crate::SliceShuffleError(()))
        } else {
            let len = src.len();
            (0..len - 1).for_each(|i| {
                let j = self.nextu().to_index(len);
                src.swap(i, j);
            });
            Ok(())
        }
    }

    /// Returns an iterator over `src` in a shuffled order.
    #[inline(always)]
    fn shuffled<'a, T>(
        &mut self,
        src: &'a [T],
    ) -> crate::ShuffleResult<crate::ShuffledIter<'a, T>> {
        if src.is_empty() {
            return Err(crate::SliceShuffleError(()));
        }
        let len = src.len();
        let mut indices: Vec<usize> = (0..len).collect();
        (0..len - 1).for_each(|i| {
            let j = self.nextu().to_index(len);
            indices.swap(i, j);
        });
        Ok(crate::ShuffledIter {
            slice: src,
            indices,
            pos: 0,
        })
    }
}

impl<R: Rng + ?Sized> Shuffle for R {}
