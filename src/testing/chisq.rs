//! Chi-squared uniformity test harness for RNGs.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine: the test
//! math and `Config`/`Result`/`Error`/`Verdict` types live in `cribler`, this
//! module only supplies the `[0, 1)` sampler closures.

use crate::rng::{Rng32, Rng64};
use cribler::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;

pub use cribler::{ChiSqConfig, ChiSqError, ChiSqResult, ChiSqVerdict};

struct ChiSqCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

macro_rules! impl_chisq_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Chi-squared uniformity tester for `Rng", $bits, "` generators.")]
            #[doc = ""]
            #[doc = "Wraps a mutable reference to a generator and runs a chi-squared test over the"]
            #[doc = "uniform `[0, 1)` floats derived from its `nextu` output."]
            pub struct [<ChiSq $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: ChiSqConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<ChiSq $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: ChiSqConfig::default(),
                    }
                }

                #[deprecated(note = "use `from_urng` instead")]
                pub fn new(rng: &'a mut R) -> Self {
                    Self::from_urng(rng)
                }

                pub fn with_config(rng: &'a mut R, config: ChiSqConfig) -> Result<Self, ChiSqError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> ChiSqConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: ChiSqConfig) -> Result<(), ChiSqError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<ChiSqResult, ChiSqError> {
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    cribler::run_chisq(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<ChiSq $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `ChiSq", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: ChiSqConfig,
                ) -> Result<Self, ChiSqError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<ChiSqSuite $bits>]<'a> {
                config: ChiSqConfig,
                cases: Vec<ChiSqCase<'a>>,
            }

            impl<'a> Default for [<ChiSqSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: ChiSqConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<ChiSqSuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: ChiSqConfig) -> Result<Self, ChiSqError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> ChiSqConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: ChiSqConfig) -> Result<(), ChiSqError> {
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
                ) -> Result<&mut Self, ChiSqError> {
                    let name = name.into();
                    self.cases.push(ChiSqCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, ChiSqError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(ChiSqCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<ChiSqResult>, ChiSqError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(ChiSqError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_chisq(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_chisq_for_rng!(32);
impl_chisq_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn chisq32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut chisq = ChiSq32::from_urng(&mut rng);
        let res = unsafe { chisq.run(stringify!(Sfmt19937)).unwrap_unchecked() };
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }

    #[test]
    fn chisq64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut chisq = ChiSq64::from_urng(&mut rng);
        let res = unsafe { chisq.run(stringify!(Sfmt1993764)).unwrap_unchecked() };
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }
}
