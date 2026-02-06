use crate::rng::Rng64;

/// Randomly selects an index based on weights using a binary search approach.
///
/// # Examples
///
/// ```
/// use urng::bst::search;
/// use urng::rng64::Sfc64;
///
/// let mut rng = Sfc64::new(1);
/// let index = search(&mut rng, &[1.0, 9.0]);
/// assert!(index == Some(0) || index == Some(1));
/// ```
pub fn search<R: Rng64>(rng: &mut R, weights: &[f64]) -> Option<usize> {
    if weights.is_empty() {
        None
    } else {
        let mut total_weight: f64 = 0.0;
        let mut accumulate_weights = Vec::with_capacity(weights.len());
        let length = weights.len();

        for weight in weights.iter() {
            total_weight += weight;
            accumulate_weights.push(total_weight);
        }

        let point = rng.randf(0.0, total_weight);
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
                    return Some(middle);
                }
                top = middle;
            }
        }
        Some(top)
    }
}

/// Randomly selects an item based on weights using a binary search approach.
///
/// # Examples
///
/// ```
/// use urng::bst::choice;
/// use urng::rng64::Sfc64;
///
/// let mut rng = Sfc64::new(1);
/// let items = ["a", "b"];
/// let index = choice(&mut rng, &[1.0, 9.0], &items);
/// assert!(index == Some(&"a") || index == Some(&"b"));
/// ```
pub fn choice<'a, R: Rng64, T>(rng: &mut R, weights: &[f64], items: &'a [T]) -> Option<&'a T> {
    if weights.is_empty() || items.is_empty() || weights.len() != items.len() {
        None
    } else {
        Some(&items[search(rng, weights)?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng64::Mt1993764;

    #[test]
    fn search_works() {
        let mut rng = Mt1993764::new(1);
        let weights = [1.0, 9.0];
        let index = search(&mut rng, &weights);
        assert!(index == Some(0) || index == Some(1));
    }

    #[test]
    fn choice_works() {
        let mut rng = Mt1993764::new(1);
        let weights = [1.0, 9.0];
        let items = ["a", "b"];
        let item = choice(&mut rng, &weights, &items);
        assert!(item == Some(&"a") || item == Some(&"b"));
    }
}
