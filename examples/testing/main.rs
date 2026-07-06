use rand::SeedableRng;
use urng::testing::{ChiSq32, McPi32};

fn main() -> anyhow::Result<()> {
    {
        let mut rng = urng::Sfc32::new(0);

        let mut chi = ChiSq32::from_urng(&mut rng);
        let chi_result = chi.run("urng.chisq")?;
        println!("{:?}", chi_result);

        let mut mc = McPi32::from_urng(&mut rng);
        let mi_result = mc.run("urng.mcpi")?;
        println!("{:?}", mi_result);
    }

    {
        let mut rng = rand_sfc::Sfc32::seed_from_u64(0);

        let mut chi = ChiSq32::from_rand(&mut rng);
        let chi_result = chi.run("rand.chisq")?;
        println!("{:?}", chi_result);

        let mut mc = McPi32::from_rand(&mut rng);
        let mi_result = mc.run("rand.mcpi")?;
        println!("{:?}", mi_result);
    }

    Ok(())
}
