use urng::{Choice, Philox32x4, Philox64, Rng, Sfc32, Shuffle};

fn main() -> anyhow::Result<()> {
    let mut rng = Sfc32::default();
    println!("{}", rng.nextu());
    println!("{}", rng.nextf());

    let mut rng = Philox32x4::default();
    println!("{}", rng.nextu());
    println!("{}", rng.nextf());

    let mut rng = Philox64::default();
    println!("{}", rng.nextu());
    println!("{}", rng.nextf());

    let items = [0u32; 8];
    println!("items: {:?}", items);

    let rand: Vec<_> = items.iter().map(|_| rng.nextu()).collect();
    println!("randomised: {:?}", rand);

    let shuffled: Vec<_> = rng.shuffled(&rand)?.collect();
    println!("shuffled: {:?}", shuffled);

    let mut shuffle: Vec<_> = shuffled.iter().copied().collect();
    rng.shuffle(&mut shuffle)?;

    let choiced = rng.choice_mut(&mut shuffle);
    println!("choiced: {}", choiced);

    Ok(())
}
