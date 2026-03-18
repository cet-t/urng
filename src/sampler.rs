use crate::rng::{Rng32, Rng64};

/// Weighted random sampling trait for 32-bit generators.
///
/// Implementors provide O(1) or O(log n) weighted index selection.
/// Available implementations: [`crate::sampler32::Bst32`], [`crate::sampler32::Alias32`].
///
/// # Examples
///
/// ```
/// use urng::sampler::Sampler32;
/// use urng::sampler32::Alias32;
/// use urng::rng32::Mt19937;
///
/// let mut rng = Mt19937::new(1);
/// let mut sampler = Alias32::new(&mut rng, &[1.0f32, 9.0]);
/// assert_eq!(sampler.sample(), 1);
/// ```
pub trait Sampler32<'a, R: Rng32 + 'a> {
    /// Creates a new sampler with the given random number generator and weights.
    fn new(rng: &'a mut R, weights: &[f32]) -> Self;
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[f32]);
}

/// Weighted random sampling trait for 64-bit generators.
///
/// Implementors provide O(1) or O(log n) weighted index selection.
/// Available implementations: [`crate::sampler64::Bst64`], [`crate::sampler64::Alias64`].
///
/// # Examples
///
/// ```
/// use urng::sampler::Sampler64;
/// use urng::sampler64::Alias64;
/// use urng::rng64::Mt1993764;
///
/// let mut rng = Mt1993764::new(1);
/// let mut sampler = Alias64::new(&mut rng, &[1.0f64, 2.0, 4.0, 8.0]);
/// assert_eq!(sampler.sample(), 0);
/// ```
pub trait Sampler64<'a, R: Rng64 + 'a> {
    /// Creates a new sampler with the given random number generator and weights.
    fn new(rng: &'a mut R, weights: &[f64]) -> Self;
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[f64]);
}
