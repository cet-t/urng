use crate::rng::Rng32;
use crate::sampler::Sampler32;

/// Weighted sampler using Walker's Alias Method (O(1) sample, O(n) build).
///
/// Preferred over [`Bst32`](super::Bst32) when sampling repeatedly from the same weight distribution.
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
///
/// // Update weights
/// sampler.weights(&[5.0, 5.0]);
/// assert!(sampler.sample() < 2);
/// ```
#[derive(Debug)]
pub struct Alias32<'a, R: Rng32 + 'a> {
    rng: &'a mut R,
    prob: Vec<f32>,
    alias: Vec<usize>,
}

impl<'a, R: Rng32 + 'a> Alias32<'a, R> {
    fn build(weights: &[f32]) -> (Vec<f32>, Vec<usize>) {
        let n = weights.len();
        let total: f32 = weights.iter().sum();
        let scale = n as f32 / total;

        let mut prob = vec![0.0f32; n];
        let mut alias = vec![0usize; n];
        let mut small = Vec::with_capacity(n);
        let mut large = Vec::with_capacity(n);
        let mut p: Vec<f32> = weights.iter().map(|&w| w * scale).collect();

        for (i, &pi) in p.iter().enumerate() {
            if pi < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while let (Some(s), Some(l)) = (small.pop(), large.last().copied()) {
            prob[s] = p[s];
            alias[s] = l;
            p[l] -= 1.0 - p[s];
            if p[l] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        for i in large {
            prob[i] = 1.0;
        }
        for i in small {
            prob[i] = 1.0;
        }

        (prob, alias)
    }
}

impl<'a, R: Rng32 + 'a> Sampler32<'a, R> for Alias32<'a, R> {
    fn new(rng: &'a mut R, weights: &[f32]) -> Self {
        let (prob, alias) = Self::build(weights);
        Self { rng, prob, alias }
    }

    fn weights(&mut self, weights: &[f32]) {
        let (prob, alias) = Self::build(weights);
        self.prob = prob;
        self.alias = alias;
    }

    fn sample(&mut self) -> usize {
        let n = self.prob.len();
        let i = self.rng.randi(0, n as i32 - 1) as usize;
        let u = self.rng.randf(0.0, 1.0);
        if u < self.prob[i] { i } else { self.alias[i] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng32::Mt19937;
    use crate::sampler::Sampler32;

    #[test]
    fn alias32_works() {
        let mut rng = Mt19937::new(1);
        let mut sampler = Alias32::new(&mut rng, &[1.0f32, 9.0f32]);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
    }
}
