//! Runs test harness for RNGs: detects sequential correlation by analyzing
//! the lengths of monotonic ascending/descending runs in the output sequence.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine.

use crate::rng::{Rng32, Rng64};
use cribler::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;

pub use cribler::{RunsConfig, RunsError, RunsResult, RunsVerdict};

struct RunsCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

macro_rules! impl_runs_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Runs (up/down monotonic run) tester for `Rng", $bits, "` generators.")]
            pub struct [<Runs $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: RunsConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<Runs $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: RunsConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: RunsConfig) -> Result<Self, RunsError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> RunsConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: RunsConfig) -> Result<(), RunsError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<RunsResult, RunsError> {
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    cribler::run_runs(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<Runs $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `Runs", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: RunsConfig,
                ) -> Result<Self, RunsError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<RunsSuite $bits>]<'a> {
                config: RunsConfig,
                cases: Vec<RunsCase<'a>>,
            }

            impl<'a> Default for [<RunsSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: RunsConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<RunsSuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: RunsConfig) -> Result<Self, RunsError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> RunsConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: RunsConfig) -> Result<(), RunsError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn len(&self) -> usize {
                    self.cases.len()
                }

                pub fn is_empty(&self) -> bool {
                    self.cases.is_empty()
                }

                pub fn clear(&mut self) {
                    self.cases.clear();
                }

                pub fn add_rng<R: [<Rng $bits>] + 'a>(
                    &mut self,
                    name: impl Into<String>,
                    rng: &'a mut R,
                ) -> Result<&mut Self, RunsError> {
                    let name = name.into();
                    self.cases.push(RunsCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, RunsError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(RunsCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<RunsResult>, RunsError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(RunsError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_runs(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_runs_for_rng!(32);
impl_runs_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn runs32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut runs = Runs32::from_urng(&mut rng);
        let res = runs.run(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, RunsVerdict::Pass);
    }

    #[test]
    fn runs64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut runs = Runs64::from_urng(&mut rng);
        let res = runs.run(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, RunsVerdict::Pass);
    }
}
