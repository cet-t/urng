//! "Paranoid" meta-test harness: elevates any single statistical test into a
//! battery-level test, following the "Testing Strategy and Result
//! Interpretation" guidance of NIST SP 800-22 rev. 1a, §4.2.
//!
//! Fully re-exported from [`cribler`]: this module is bit-width-agnostic
//! (trial closures are `FnMut(usize) -> f64`, independent of any `Rng32`/
//! `Rng64` generator), so there is nothing urng-specific to wrap. Wrap a call
//! to [`crate::testing::ChiSq32`], [`crate::testing::Nist32`], or any other
//! test in a closure that returns a p-value, and [`run_paranoid`] (or
//! [`ParanoidSuite`] for batching named cases) applies the meta-level checks.

pub use cribler::{
    ParanoidConfig, ParanoidError, ParanoidResult, ParanoidSuite, ParanoidVerdict, p_value_from_z,
    run_paranoid,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::testing::chisq::ChiSq32;

    #[test]
    fn paranoid_passes_for_good_generator() {
        let config = ParanoidConfig {
            trials: 30,
            ..ParanoidConfig::default()
        };
        let mut trial = |i: usize| {
            let mut rng = Sfmt19937::new(i as u64 + 1);
            let z = ChiSq32::from_urng(&mut rng)
                .run(format!("trial-{i}"))
                .unwrap()
                .z_score;
            p_value_from_z(z)
        };
        let res = run_paranoid("Sfmt19937.paranoid_chisq", config, &mut trial).unwrap();
        assert_eq!(res.verdict, ParanoidVerdict::Pass);
    }

    #[test]
    fn paranoid_rejects_biased_generator() {
        let config = ParanoidConfig {
            trials: 20,
            ..ParanoidConfig::default()
        };
        // A trial closure that always reports a maximally-suspicious p-value
        // must fail both the proportion and uniformity checks.
        let mut trial = |_i: usize| 0.0f64;
        let res = run_paranoid("always-fail", config, &mut trial).unwrap();
        assert_eq!(res.verdict, ParanoidVerdict::Fail);
    }
}
