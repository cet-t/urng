//! Convenience trait to run statistical tests directly on an RNG instance,
//! without manually constructing a [`crate::testing::ChiSq32`]/[`crate::testing::McPi32`] harness.

use crate::rng::{Rng32, Rng64};
use crate::testing::birthday::{
    Birthday32, Birthday64, BirthdayConfig, BirthdayError, BirthdayResult,
};
use crate::testing::chisq::{ChiSq32, ChiSq64, ChiSqConfig, ChiSqError, ChiSqResult};
use crate::testing::ks::{Ks32, Ks64, KsConfig, KsError, KsResult};
use crate::testing::mcpi::{McPi32, McPi64, McPiConfig, McPiError, McPiResult};
use crate::testing::nist::{Nist32, Nist64, NistConfig, NistError, NistResult};
use crate::testing::runs::{Runs32, Runs64, RunsConfig, RunsError, RunsResult};
use crate::testing::serial::{Serial32, Serial64, SerialConfig, SerialError, SerialResult};

macro_rules! impl_test_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Adds `run_chisq`/`run_mcpi` helpers to any ", $bits, "-bit RNG.")]
            ///
            /// # Examples
            ///
            /// ```
            /// use urng::*;
            /// use urng::testing::{ChiSqVerdict, Test32};
            ///
            /// let mut rng = Sfc32::new(0);
            /// let result = rng.run_chisq("sfc32").unwrap();
            /// assert_eq!(result.verdict, ChiSqVerdict::Pass);
            /// ```
            pub trait [<Test $bits>]: [<Rng $bits>] + Sized {
                #[doc = concat!("Runs a chi-squared uniformity test on `self` with the default [`ChiSqConfig`].")]
                fn run_chisq(&mut self, name: impl Into<String>) -> Result<ChiSqResult, ChiSqError> {
                    [<ChiSq $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a chi-squared uniformity test on `self` with a custom [`ChiSqConfig`].")]
                fn run_chisq_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: ChiSqConfig,
                ) -> Result<ChiSqResult, ChiSqError> {
                    [<ChiSq $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs a Monte Carlo estimation of \u{3c0} on `self` with the default [`McPiConfig`].")]
                fn run_mcpi(&mut self, name: impl Into<String>) -> Result<McPiResult, McPiError> {
                    [<McPi $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a Monte Carlo estimation of \u{3c0} on `self` with a custom [`McPiConfig`].")]
                fn run_mcpi_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: McPiConfig,
                ) -> Result<McPiResult, McPiError> {
                    [<McPi $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs a serial (multi-dimensional uniformity) test on `self` with the default [`SerialConfig`].")]
                fn run_serial(&mut self, name: impl Into<String>) -> Result<SerialResult, SerialError> {
                    [<Serial $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a serial test on `self` with a custom [`SerialConfig`].")]
                fn run_serial_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: SerialConfig,
                ) -> Result<SerialResult, SerialError> {
                    [<Serial $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs a runs (monotonic run length) test on `self` with the default [`RunsConfig`].")]
                fn run_runs(&mut self, name: impl Into<String>) -> Result<RunsResult, RunsError> {
                    [<Runs $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a runs test on `self` with a custom [`RunsConfig`].")]
                fn run_runs_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: RunsConfig,
                ) -> Result<RunsResult, RunsError> {
                    [<Runs $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs a Kolmogorov-Smirnov uniformity test on `self` with the default [`KsConfig`].")]
                fn run_ks(&mut self, name: impl Into<String>) -> Result<KsResult, KsError> {
                    [<Ks $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a Kolmogorov-Smirnov test on `self` with a custom [`KsConfig`].")]
                fn run_ks_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: KsConfig,
                ) -> Result<KsResult, KsError> {
                    [<Ks $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs a birthday spacing test on `self` with the default [`BirthdayConfig`].")]
                fn run_birthday(&mut self, name: impl Into<String>) -> Result<BirthdayResult, BirthdayError> {
                    [<Birthday $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs a birthday spacing test on `self` with a custom [`BirthdayConfig`].")]
                fn run_birthday_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: BirthdayConfig,
                ) -> Result<BirthdayResult, BirthdayError> {
                    [<Birthday $bits>]::with_config(self, config)?.run(name)
                }

                #[doc = concat!("Runs the NIST SP 800-22 bit-level test suite on `self` with the default [`NistConfig`].")]
                fn run_nist(&mut self, name: impl Into<String>) -> Result<NistResult, NistError> {
                    [<Nist $bits>]::from_urng(self).run(name)
                }

                #[doc = concat!("Runs the NIST SP 800-22 suite on `self` with a custom [`NistConfig`].")]
                fn run_nist_with_config(
                    &mut self,
                    name: impl Into<String>,
                    config: NistConfig,
                ) -> Result<NistResult, NistError> {
                    [<Nist $bits>]::with_config(self, config)?.run(name)
                }
            }

            impl<T: [<Rng $bits>]> [<Test $bits>] for T {}
        }
    };
}

impl_test_for_rng!(32);
impl_test_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::chisq::ChiSqVerdict;
    use crate::testing::mcpi::McPiVerdict;
    use crate::{Sfmt19937, Sfmt1993764};

    #[test]
    fn test32_run_chisq_works() {
        let mut rng = Sfmt19937::new(0);
        let res = rng.run_chisq(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }

    #[test]
    fn test32_run_mcpi_works() {
        let mut rng = Sfmt19937::new(0);
        let res = rng.run_mcpi(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }

    #[test]
    fn test64_run_chisq_works() {
        let mut rng = Sfmt1993764::new(0);
        let res = rng.run_chisq(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }

    #[test]
    fn test64_run_mcpi_works() {
        let mut rng = Sfmt1993764::new(0);
        let res = rng.run_mcpi(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }

    #[test]
    fn test32_run_serial_runs_ks_birthday_work() {
        use crate::testing::birthday::BirthdayVerdict;
        use crate::testing::ks::KsVerdict;
        use crate::testing::runs::RunsVerdict;
        use crate::testing::serial::SerialVerdict;

        let mut rng = Sfmt19937::new(0);
        assert_eq!(
            rng.run_serial(stringify!(Sfmt19937)).unwrap().verdict,
            SerialVerdict::Pass
        );
        assert_eq!(
            rng.run_runs(stringify!(Sfmt19937)).unwrap().verdict,
            RunsVerdict::Pass
        );
        assert_eq!(
            rng.run_ks(stringify!(Sfmt19937)).unwrap().verdict,
            KsVerdict::Pass
        );
        assert_eq!(
            rng.run_birthday(stringify!(Sfmt19937)).unwrap().verdict,
            BirthdayVerdict::Pass
        );
    }

    /// Regression test: Philox32x4/Threefry32x4/Threefry32x2/Philox64 now
    /// implement Rng32/Rng64 directly (internal ring buffer over their
    /// block-generation), so they participate in the shared Test32/Test64
    /// harness with no special-casing.
    #[test]
    fn block_generators_work_with_test_harness() {
        use crate::rng64::Philox64;
        use crate::{Philox32x4, Threefry32x2, Threefry32x4};

        let mut rng = Philox32x4::new(0);
        assert_eq!(
            rng.run_chisq("Philox32x4").unwrap().verdict,
            ChiSqVerdict::Pass
        );

        let mut rng = Threefry32x4::new(0);
        assert_eq!(
            rng.run_chisq("Threefry32x4").unwrap().verdict,
            ChiSqVerdict::Pass
        );

        let mut rng = Threefry32x2::new(0);
        assert_eq!(
            rng.run_chisq("Threefry32x2").unwrap().verdict,
            ChiSqVerdict::Pass
        );

        let mut rng = Philox64::new(0);
        assert_eq!(
            rng.run_chisq("Philox64").unwrap().verdict,
            ChiSqVerdict::Pass
        );
    }
}
