use crate::rng::Rng32;
use crate::sampler::Sampler32;

/// Weighted sampler using cumulative sums and binary search (O(log n) sample, O(n) build).
///
/// # Examples
///
/// ```
/// use urng::sampler::Sampler32;
/// use urng::sampler32::Bst32;
/// use urng::rng32::Mt19937;
///
/// let mut rng = Mt19937::new(1);
/// let mut sampler = Bst32::new(&mut rng, &[1.0f32, 9.0]);
/// assert_eq!(sampler.sample(), 1);
/// ```
#[derive(Debug)]
pub struct Bst32<'a, R: Rng32 + 'a> {
    rng: &'a mut R,
    cumulative: Vec<f32>,
}

impl<'a, R: Rng32 + 'a> Bst32<'a, R> {
    /// Creates a new sampler with the given random number generator and weights.
    pub fn new(rng: &'a mut R, weights: &[f32]) -> Self {
        Self {
            rng,
            cumulative: Self::build_cumulative(weights),
        }
    }

    fn build_cumulative(weights: &[f32]) -> Vec<f32> {
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut sum = 0.0f32;
        for &w in weights {
            sum += w;
            cumulative.push(sum);
        }
        cumulative
    }
}

impl<'a, R: Rng32 + 'a> Sampler32<'a, R> for Bst32<'a, R> {
    fn weights(&mut self, weights: &[f32]) {
        self.cumulative = Self::build_cumulative(weights);
    }

    fn sample(&mut self) -> usize {
        let total = *self.cumulative.last().unwrap_or(&0.0);
        let r = self.rng.randf(0.0, total);
        match self
            .cumulative
            .binary_search_by(|&c| c.partial_cmp(&r).unwrap())
        {
            Ok(i) | Err(i) => i.min(self.cumulative.len() - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng32::Mt19937;
    use crate::sampler::Sampler32;

    #[test]
    fn bst32_works() {
        let mut rng = Mt19937::new(1);
        let mut sampler = Bst32::new(&mut rng, &[1.0f32, 9.0f32]);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 0);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
    }
}
