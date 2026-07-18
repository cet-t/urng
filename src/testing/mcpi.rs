//! Monte Carlo pi-estimation test harness for RNGs.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine.

use crate::rng::{Rng32, Rng64};
use cribler::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;

pub use cribler::{McPiConfig, McPiError, McPiResult, McPiVerdict};

struct McPiCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

macro_rules! impl_mcpi_for_rng {
    ($bits:expr) => {
        paste::paste!{
            #[doc = concat!("Monte Carlo estimation of \u{3c0} using ", $bits, "-bit RNGs.")]
            #[doc = ""]
            #[doc = "Wraps a mutable reference to a generator and counts how many of its `[0, 1)`"]
            #[doc = "point pairs fall inside the unit quarter-circle to estimate \u{3c0}."]
            pub struct [<McPi $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: McPiConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<McPi $bits>]<'a, R> {
                #[doc = concat!("Creates a tester for `", stringify!([<Rng $bits>]), "` using the default configuration.")]
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: McPiConfig::default(),
                    }
                }

                #[deprecated(note = "use `from_urng` instead")]
                #[doc = "Deprecated alias for [`from_urng`](Self::from_urng)."]
                pub fn new(rng: &'a mut R) -> Self {
                    Self::from_urng(rng)
                }

                #[doc = "Creates a tester with an explicit, validated configuration."]
                pub fn with_config(rng: &'a mut R, config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                #[doc = "Returns the current configuration."]
                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                #[doc = "Replaces the configuration after validating it."]
                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                #[doc = "Runs the Monte Carlo test, returning a [`McPiResult`] (or the first validation error)."]
                pub fn run(&mut self, name: impl Into<String>) -> Result<McPiResult, McPiError> {
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    cribler::run_mcpi(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<McPi $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `McPi", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                #[doc = "Creates a tester with an explicit configuration from any `rand_core::Rng`."]
                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: McPiConfig,
                ) -> Result<Self, McPiError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            #[doc = concat!("A suite that runs multiple Monte Carlo \u{3c0} test cases and collects their [`McPiResult`]s.")]
            pub struct [<McPiSuite $bits>]<'a> {
                config: McPiConfig,
                cases: Vec<McPiCase<'a>>,
            }

            impl<'a> Default for [<McPiSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: McPiConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<McPiSuite $bits>]<'a> {
                #[doc = "Creates an empty suite with the default configuration."]
                pub fn new() -> Self {
                    Self::default()
                }

                #[doc = "Creates an empty suite with an explicit, validated configuration."]
                pub fn with_config(config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                #[doc = "Returns the current configuration."]
                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                #[doc = "Replaces the configuration after validating it."]
                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                #[doc = "Returns the number of registered cases."]
                pub fn len(&self) -> usize {
                    self.cases.len()
                }

                #[doc = "Returns `true` if no cases are registered."]
                pub fn is_empty(&self) -> bool {
                    self.cases.is_empty()
                }

                #[doc = "Removes all registered cases."]
                pub fn clear(&mut self) {
                    self.cases.clear();
                }

                #[doc = concat!("Registers a `", stringify!([<Rng $bits>]), "` generator as a named test case.")]
                pub fn add_rng<R: [<Rng $bits>] + 'a>(
                    &mut self,
                    name: impl Into<String>,
                    rng: &'a mut R,
                ) -> Result<&mut Self, McPiError> {
                    let name = name.into();
                    self.cases.push(McPiCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                #[doc = "Registers an arbitrary `[0, 1)` sampler closure as a named test case."]
                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, McPiError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(McPiCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                #[doc = "Runs every case, returning a [`McPiResult`] per case (or the first error)."]
                pub fn run(&mut self) -> Result<Vec<McPiResult>, McPiError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(McPiError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_mcpi(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_mcpi_for_rng!(32);
impl_mcpi_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn mcpi32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut mcpi = McPi32::from_urng(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt19937)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }

    #[test]
    fn mcpi64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut mcpi = McPi64::from_urng(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt1993764)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }
}
