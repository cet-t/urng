use urng::{Rng, Sfc64};

fn main() -> anyhow::Result<()> {
    let mut mt = Sfc64::default();
    for _ in 0..10 {
        println!("{}", mt.randi(0, 10));
    }

    Ok(())
}
