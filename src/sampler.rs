//! Weighted random selection.

pub mod b32;
pub mod b64;

pub use b32::*;
pub use b64::*;

use crate::rng::{Rng, Word};

/// Weighted random sampling trait, generic over the generator's [`Word`].
///
/// Implementors provide O(1) or O(log n) weighted index selection.
/// Available implementations: [`crate::sampler::b32::Bst32`], [`crate::sampler::b32::Alias32`],
/// [`crate::sampler::b64::Bst64`], [`crate::sampler::b64::Alias64`].
///
/// # Examples
///
/// ```
/// use urng::{Sampler, Alias32, Mt19937};
///
/// let mut rng = Mt19937::new(1);
/// let mut sampler = Alias32::new(&mut rng, &[1.0f32, 9.0]);
/// assert!(sampler.sample() < 2);
/// ```
pub trait Sampler<'a, R: Rng + 'a> {
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[<R::Word as Word>::Float]);
}
