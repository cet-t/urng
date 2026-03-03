use crate::rng::Rng64;
use crate::sampler::Sampler64;

#[derive(Debug)]
pub struct Bst64<'a, R: Rng64 + 'a> {
    rng: &'a mut R,
    weights: Vec<f64>,
}

impl<'a, R: Rng64 + 'a> Sampler64<'a, R> for Bst64<'a, R> {
    fn new(rng: &'a mut R, weights: &[f64]) -> Self {
        Self {
            rng,
            weights: weights.to_vec(),
        }
    }

    fn weights(&mut self, weights: &[f64]) {
        self.weights.clear();
        self.weights.extend_from_slice(weights);
    }

    fn sample(&mut self) -> usize {
        let mut sum = 0.0;
        for w in &self.weights {
            sum += w;
        }
        let r = self.rng.randf(0.0, sum);
        let mut current = 0.0;
        for (i, w) in self.weights.iter().enumerate() {
            current += w;
            if current >= r {
                return i;
            }
        }
        self.weights.len() - 1
    }
}
