//! Weighted random selection.

pub mod b32;
pub mod b64;

pub use b32::*;
pub use b64::*;

use crate::rng::Rng;

/// Weighted random sampling trait for 32-bit generators.
///
/// Implementors provide O(1) or O(log n) weighted index selection.
/// Available implementations: [`crate::sampler::b32::Bst32`], [`crate::sampler::b32::Alias32`].
///
/// # Examples
///
/// ```
/// use urng::{Sampler32, Alias32, Mt19937};
///
/// let mut rng = Mt19937::new(1);
/// let mut sampler = Alias32::new(&mut rng, &[1.0f32, 9.0]);
/// assert!(sampler.sample() < 2);
/// ```
pub trait Sampler32<'a, R: Rng<Word = u32> + 'a> {
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[f32]);
}

/// Weighted random sampling trait for 64-bit generators.
///
/// Implementors provide O(1) or O(log n) weighted index selection.
/// Available implementations: [`crate::sampler::b64::Bst64`], [`crate::sampler::b64::Alias64`].
///
/// # Examples
///
/// ```
/// use urng::{Sampler64, Alias64, Mt1993764};
///
/// let mut rng = Mt1993764::new(1);
/// let mut sampler = Alias64::new(&mut rng, &[1.0f64, 2.0, 4.0, 8.0]);
/// assert!(sampler.sample() < 4);
/// ```
pub trait Sampler64<'a, R: Rng<Word = u64> + 'a> {
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[f64]);
}
