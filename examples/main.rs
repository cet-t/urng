use anyhow::Result;
use std::collections::HashMap;
use urng::{choice, rng64::Mt1993764};

fn bst_test() -> Result<()> {
    const N: usize = 100_000;
    let weights = [0.01, 1.0, 20.0, 80.0];
    let items = ["SSR", "SR", "R", "N"];
    let mut rng = Mt1993764::new(1, 256);

    let mut results = HashMap::<&str, i32>::from_iter(items.iter().map(|&k| (k, 0)));
    for _ in 0..N {
        *results
            .entry(choice!(&mut rng, weights, items).unwrap())
            .or_insert(0) += 1;
    }

    let mut results = results.iter().collect::<Vec<_>>();
    results.sort_by_key(|(_, v)| -**v);
    results.iter().for_each(|(k, v)| {
        let per = **v as f64 * 100.0 / N as f64;
        println!("{:<3}: {:>6.2}%", k, per);
    });

    Ok(())
}

fn main() -> Result<()> {
    bst_test()?;
    Ok(())
}
