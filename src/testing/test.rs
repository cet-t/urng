//! Convenience trait to run statistical tests directly on an RNG instance,
//! without manually constructing a [`crate::testing::ChiSq32`]/[`crate::testing::McPi32`] harness.

use crate::rng::{Rng32, Rng64};
use crate::testing::chisq::{ChiSq32, ChiSq64, ChiSqConfig, ChiSqError, ChiSqResult};
use crate::testing::mcpi::{McPi32, McPi64, McPiConfig, McPiError, McPiResult};

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
}
