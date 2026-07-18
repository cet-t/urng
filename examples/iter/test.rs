use urng::{Choice32, Sfc32, Shuffle32};

fn main() -> anyhow::Result<()> {
    let mut vals: Vec<_> = (0..10).collect();
    println!("{:?}", vals);

    let mut rng = Sfc32::default();
    let done: Vec<_> = rng.shuffled(&vals)?.collect();
    println!("{:?}", done);

    let _ = rng.shuffle(&mut vals);
    println!("{:?}", vals);

    println!("{}", rng.choice(&vals));

    Ok(())
}
