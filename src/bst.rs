use crate::rng::Rng64;

/// A weighted random selection structure using a Binary Search Tree (BST) approach.
///
/// # Examples
///
/// ```
/// use rng_pack::bst::Bst;
/// use rng_pack::rng64::Xorshift64;
///
/// let choices = vec!["a", "b", "c"];
/// let weights = vec![1.0, 2.0, 7.0];
/// let rng = Xorshift64::new(1);
/// let mut bst = Bst::new(choices, weights, rng);
///
/// let result = bst.choice();
/// ```
pub struct Bst<T, R: Rng64> {
    source: Vec<T>,
    weights: Vec<f64>,
    rng: R,
}

impl<T, R: Rng64> Bst<T, R> {
    /// Creates a new `Bst` instance.
    ///
    /// # Panics
    ///
    /// Panics if `source` and `weights` have different lengths.
    pub fn new(source: Vec<T>, weights: Vec<f64>, rng: R) -> Self {
        if source.len() != weights.len() {
            panic!("Source and weights must have the same length");
        }
        Self {
            source,
            weights,
            rng,
        }
    }

    /// Randomly selects an index based on weights using a binary search approach.
    ///
    /// # Examples
    ///
    /// ```
    /// use rng_pack::bst::Bst;
    /// use rng_pack::rng64::Xorshift64;
    ///
    /// let mut bst = Bst::new(vec!["a", "b"], vec![1.0, 9.0], Xorshift64::new(1));
    /// let index = bst.search();
    /// assert!(index == 0 || index == 1);
    /// ```
    pub fn search(&mut self) -> usize {
        if self.source.is_empty() {
            0
        } else {
            let mut total_weight: f64 = 0.0;
            let mut accumulate_weights = Vec::new();
            let length = self.weights.len();

            for weight in self.weights.iter() {
                total_weight += weight;
                accumulate_weights.push(total_weight);
            }

            let point = self.rng.randf(0.0, total_weight);
            let mut bottom = 0;
            let mut top = length - 1;
            while bottom < top {
                let middle = (bottom + top) / 2;
                if point > accumulate_weights[middle] {
                    bottom = middle + 1;
                } else {
                    let p = if middle > 0 {
                        accumulate_weights[middle - 1]
                    } else {
                        0.0
                    };
                    if point >= p {
                        return middle;
                    }
                    top = middle;
                }
            }
            top
        }
    }

    /// Randomly selects an element from the source based on weights.
    ///
    /// # Examples
    ///
    /// ```
    /// use rng_pack::bst::Bst;
    /// use rng_pack::rng64::Xorshift64;
    ///
    /// let mut bst = Bst::new(vec!["a", "b"], vec![1.0, 9.0], Xorshift64::new(1));
    /// let val = bst.choice();
    /// assert!(*val == "a" || *val == "b");
    /// ```
    pub fn choice(&mut self) -> &T {
        let index = self.search();
        &self.source[index]
    }
}
