use crate::rng::Rng64;
use crate::sampler::Sampler64;

// --- Bst64 ---

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
/// assert_eq!(sampler.sample(), 3);
/// ```
#[derive(Debug)]
pub struct Bst64<'a, R: Rng64 + 'a> {
    rng: &'a mut R,
    cumulative: Vec<f64>,
}

impl<'a, R: Rng64 + 'a> Bst64<'a, R> {
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

impl<'a, R: Rng64 + 'a> Sampler64<'a, R> for Bst64<'a, R> {
    fn new(rng: &'a mut R, weights: &[f64]) -> Self {
        Self {
            rng,
            cumulative: Self::build_cumulative(weights),
        }
    }

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

// --- Alias64 ---

/// Weighted sampler using Walker's Alias Method (O(1) sample, O(n) build).
///
/// Preferred over [`Bst64`] when sampling repeatedly from the same weight distribution.
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
///
/// // Update weights
/// sampler.weights(&[1.0, 1.0, 1.0, 1.0]);
/// assert!(sampler.sample() < 4);
/// ```
#[derive(Debug)]
pub struct Alias64<'a, R: Rng64 + 'a> {
    rng: &'a mut R,
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl<'a, R: Rng64 + 'a> Alias64<'a, R> {
    fn build(weights: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let n = weights.len();
        let total: f64 = weights.iter().sum();
        let scale = n as f64 / total;

        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0usize; n];
        let mut small = Vec::with_capacity(n);
        let mut large = Vec::with_capacity(n);
        let mut p: Vec<f64> = weights.iter().map(|&w| w * scale).collect();

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

impl<'a, R: Rng64 + 'a> Sampler64<'a, R> for Alias64<'a, R> {
    fn new(rng: &'a mut R, weights: &[f64]) -> Self {
        let (prob, alias) = Self::build(weights);
        Self { rng, prob, alias }
    }

    fn weights(&mut self, weights: &[f64]) {
        let (prob, alias) = Self::build(weights);
        self.prob = prob;
        self.alias = alias;
    }

    fn sample(&mut self) -> usize {
        let n = self.prob.len();
        let i = self.rng.randi(0, n as i64 - 1) as usize;
        let u = self.rng.randf(0.0, 1.0);
        if u < self.prob[i] { i } else { self.alias[i] }
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
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 1);
    }

    #[test]
    fn alias64_works() {
        let mut rng = Mt1993764::new(1);
        let mut sampler = Alias64::new(&mut rng, &[1f64, 2f64, 4f64, 8f64]);
        assert_eq!(sampler.sample(), 0);
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 0);
        assert_eq!(sampler.sample(), 3);
        assert_eq!(sampler.sample(), 0);
    }
}
