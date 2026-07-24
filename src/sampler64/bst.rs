use crate::rng::Rng;
use crate::sampler::Sampler64;

/// Weighted sampler using cumulative sums and binary search (O(log n) sample, O(n) build).
///
/// # Examples
///
/// ```
/// use urng::sampler::Sampler64;
/// use urng::sampler64::Bst64;
/// use urng::rng64::Mt1993764;
///
/// let mut rng = Mt1993764::new(1);
/// let mut sampler = Bst64::new(&mut rng, &[1.0f64, 2.0, 4.0, 8.0]);
/// assert!(sampler.sample() < 4);
/// ```
#[derive(Debug)]
pub struct Bst64<'a, R: Rng<Word = u64> + 'a> {
    rng: &'a mut R,
    cumulative: Vec<f64>,
}

impl<'a, R: Rng<Word = u64> + 'a> Bst64<'a, R> {
    /// Creates a new sampler with the given random number generator and weights.
    pub fn new(rng: &'a mut R, weights: &[f64]) -> Self {
        Self {
            rng,
            cumulative: Self::build_cumulative(weights),
        }
    }

    fn build_cumulative(weights: &[f64]) -> Vec<f64> {
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut sum = 0.0;
        for &w in weights {
            sum += w;
            cumulative.push(sum);
        }
        cumulative
    }
}

impl<'a, R: Rng<Word = u64> + 'a> Sampler64<'a, R> for Bst64<'a, R> {
    fn weights(&mut self, weights: &[f64]) {
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
    use crate::rng64::Mt1993764;
    use crate::sampler::Sampler64;

    #[test]
    fn bst64_works() {
        let mut rng = Mt1993764::new(1);
        let mut sampler = Bst64::new(&mut rng, &[1f64, 2f64, 4f64, 8f64]);
        // weights 1:2:4:8 (total 15) → index 3 expected ~53% of the time
        let n = 10_000;
        let mut counts = [0usize; 4];
        for _ in 0..n {
            counts[sampler.sample()] += 1;
        }
        assert!((4_800..=5_900).contains(&counts[3]), "counts = {counts:?}");
        assert!(counts[0] < counts[3], "counts = {counts:?}");
    }
}
