use rand::SeedableRng;
use urng::testing::{
    Birthday32, ChiSq32, Ks32, McPi32, Nist32, ParanoidConfig, Runs32, Serial32, p_value_from_z,
    run_paranoid,
};

fn main() -> anyhow::Result<()> {
    {
        let mut rng = urng::Sfc32::new(0);

        let mut chi = ChiSq32::from_urng(&mut rng);
        let chi_result = chi.run("urng.chisq")?;
        println!("{:?}", chi_result);

        let mut mc = McPi32::from_urng(&mut rng);
        let mi_result = mc.run("urng.mcpi")?;
        println!("{:?}", mi_result);

        let mut serial = Serial32::from_urng(&mut rng);
        let serial_result = serial.run("urng.serial")?;
        println!("{:?}", serial_result);

        let mut runs = Runs32::from_urng(&mut rng);
        let runs_result = runs.run("urng.runs")?;
        println!("{:?}", runs_result);

        let mut ks = Ks32::from_urng(&mut rng);
        let ks_result = ks.run("urng.ks")?;
        println!("{:?}", ks_result);

        let mut birthday = Birthday32::from_urng(&mut rng);
        let birthday_result = birthday.run("urng.birthday")?;
        println!("{:?}", birthday_result);

        let mut nist = Nist32::from_urng(&mut rng);
        let nist_result = nist.run("urng.nist")?;
        println!("{:?}", nist_result);
    }

    {
        let mut rng = rand_sfc::Sfc32::seed_from_u64(0);

        let mut chi = ChiSq32::from_rand(&mut rng);
        let chi_result = chi.run("rand.chisq")?;
        println!("{:?}", chi_result);

        let mut mc = McPi32::from_rand(&mut rng);
        let mi_result = mc.run("rand.mcpi")?;
        println!("{:?}", mi_result);

        let mut serial = Serial32::from_rand(&mut rng);
        let serial_result = serial.run("rand.serial")?;
        println!("{:?}", serial_result);

        let mut runs = Runs32::from_rand(&mut rng);
        let runs_result = runs.run("rand.runs")?;
        println!("{:?}", runs_result);

        let mut ks = Ks32::from_rand(&mut rng);
        let ks_result = ks.run("rand.ks")?;
        println!("{:?}", ks_result);

        let mut birthday = Birthday32::from_rand(&mut rng);
        let birthday_result = birthday.run("rand.birthday")?;
        println!("{:?}", birthday_result);

        let mut nist = Nist32::from_rand(&mut rng);
        let nist_result = nist.run("rand.nist")?;
        println!("{:?}", nist_result);
    }

    {
        // "Paranoid" meta-test: re-run the chi-squared test across 50
        // independently-seeded Sfc32 instances and check both the
        // proportion of passes and the uniformity of the resulting
        // p-values (NIST SP 800-22 §4.2).
        let config = ParanoidConfig {
            trials: 50,
            ..ParanoidConfig::default()
        };
        let mut trial = |i: usize| {
            let mut rng = urng::Sfc32::new(i as u32 + 1);
            let z = ChiSq32::from_urng(&mut rng)
                .run("paranoid-trial")
                .unwrap()
                .z_score;
            p_value_from_z(z)
        };
        let paranoid_result = run_paranoid("urng.paranoid_chisq", config, &mut trial)?;
        println!("{:?}", paranoid_result);
    }

    Ok(())
}
