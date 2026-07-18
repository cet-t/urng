//! Kolmogorov-Smirnov test harness for RNGs: checks the maximum deviation
//! between the empirical CDF of the samples and the ideal uniform CDF.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine.

use crate::rng::{Rng32, Rng64};
use cribler::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;

pub use cribler::{KsConfig, KsError, KsResult, KsVerdict};

struct KsCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

macro_rules! impl_ks_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Kolmogorov-Smirnov uniformity tester for `Rng", $bits, "` generators.")]
            pub struct [<Ks $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: KsConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<Ks $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: KsConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: KsConfig) -> Result<Self, KsError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> KsConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: KsConfig) -> Result<(), KsError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<KsResult, KsError> {
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    cribler::run_ks(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<Ks $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `Ks", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: KsConfig,
                ) -> Result<Self, KsError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<KsSuite $bits>]<'a> {
                config: KsConfig,
                cases: Vec<KsCase<'a>>,
            }

            impl<'a> Default for [<KsSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: KsConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<KsSuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: KsConfig) -> Result<Self, KsError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> KsConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: KsConfig) -> Result<(), KsError> {
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
                ) -> Result<&mut Self, KsError> {
                    let name = name.into();
                    self.cases.push(KsCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, KsError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(KsCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<KsResult>, KsError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(KsError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_ks(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_ks_for_rng!(32);
impl_ks_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn ks32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut ks = Ks32::from_urng(&mut rng);
        let res = ks.run(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, KsVerdict::Pass);
    }

    #[test]
    fn ks64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut ks = Ks64::from_urng(&mut rng);
        let res = ks.run(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, KsVerdict::Pass);
    }
}
