use crate::rng::Rng32;
use crate::sampler::Sampler32;

// --- Bst32 ---

#[derive(Debug)]
pub struct Bst32<'a, R: Rng32 + 'a> {
    rng: &'a mut R,
    cumulative: Vec<f32>,
}

impl<'a, R: Rng32 + 'a> Bst32<'a, R> {
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
    fn new(rng: &'a mut R, weights: &[f32]) -> Self {
        Self {
            rng,
            cumulative: Self::build_cumulative(weights),
        }
    }

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

// --- Alias32 ---

/// Walker's Alias Method: O(1) sample, O(n) build.
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
    fn bst32_works() {
        let mut rng = Mt19937::new(1);
        let mut sampler = Bst32::new(&mut rng, &[1.0f32, 9.0f32]);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 0);
        assert_eq!(sampler.sample(), 1);
        assert_eq!(sampler.sample(), 1);
    }

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
