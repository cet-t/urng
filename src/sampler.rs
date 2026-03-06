use crate::rng::Rng64;

pub trait Sampler64<'a, R: Rng64> {
    /// Creates a new sampler with the given random number generator and weights.
    fn new(rng: &'a mut R, weights: &[f64]) -> Self;
    /// Samples a random index based on the weights.
    fn sample(&mut self) -> usize;
    /// Updates the weights of the sampler.
    fn weights(&mut self, weights: &[f64]);
}
