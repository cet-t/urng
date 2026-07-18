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
/// Returned by [`Shuffle32::shuffled`] and [`Shuffle64::shuffled`].
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

#[macro_export]
macro_rules! impl_shuffle {
    ($bits:expr) => {
        ::paste::paste! {
            /// Shuffles a slice in-place or returns a shuffled iterator.
            pub trait [<Shuffle $bits>]: [<Rng $bits>] {
                #[inline(always)]
                fn shuffle<T>(&mut self, src: &mut [T]) -> $crate::ShuffleResult<()> {
                    if src.is_empty() {
                        Err($crate::SliceShuffleError(()))
                    } else {
                        let length = src.len() - 1;
                        (0..length).for_each(|i| {
                            let j = self.randi(0, length as [<i $bits>]) as usize;
                            src.swap(i, j);
                        });
                        Ok(())
                    }
                }

                #[inline(always)]
                fn shuffled<'a, T>(&mut self, src: &'a [T]) -> $crate::ShuffleResult<$crate::ShuffledIter<'a, T>> {
                    if src.is_empty() {
                        return Err($crate::SliceShuffleError(()));
                    }
                    let length = src.len() - 1;
                    let mut indices: Vec<usize> = (0..src.len()).collect();
                    (0..length).for_each(|i| {
                        let j = self.randi(0, length as [<i $bits>]) as usize;
                        indices.swap(i, j);
                    });
                    Ok($crate::ShuffledIter {
                        slice: src,
                        indices,
                        pos: 0,
                    })
                }
            }

            impl<T: [<Rng $bits>] + ?Sized> [<Shuffle $bits>] for T {}
        }
    };
}
