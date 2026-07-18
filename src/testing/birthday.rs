//! Birthday spacing test harness for RNGs (Marsaglia's DIEHARD "birthday
//! spacings" test): detects short periods and lattice structure by checking
//! whether collisions among sampled "birthdays" occur at the expected rate.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine.

use crate::rng::{Rng32, Rng64};
use std::collections::HashSet;

pub use cribler::{BirthdayConfig, BirthdayError, BirthdayResult, BirthdayVerdict};

struct BirthdayCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> u64 + 'a>,
}

macro_rules! impl_birthday_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Birthday spacing tester for `Rng", $bits, "` generators.")]
            pub struct [<Birthday $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: BirthdayConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<Birthday $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: BirthdayConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: BirthdayConfig) -> Result<Self, BirthdayError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> BirthdayConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: BirthdayConfig) -> Result<(), BirthdayError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<BirthdayResult, BirthdayError> {
                    let mut sampler = || self.rng.nextu() as u64;
                    cribler::run_birthday(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<Birthday $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `Birthday", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: BirthdayConfig,
                ) -> Result<Self, BirthdayError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<BirthdaySuite $bits>]<'a> {
                config: BirthdayConfig,
                cases: Vec<BirthdayCase<'a>>,
            }

            impl<'a> Default for [<BirthdaySuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: BirthdayConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<BirthdaySuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: BirthdayConfig) -> Result<Self, BirthdayError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> BirthdayConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: BirthdayConfig) -> Result<(), BirthdayError> {
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
                ) -> Result<&mut Self, BirthdayError> {
                    let name = name.into();
                    self.cases.push(BirthdayCase {
                        name,
                        sampler: Box::new(move || rng.nextu() as u64),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, BirthdayError>
                where
                    F: FnMut() -> u64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(BirthdayCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<BirthdayResult>, BirthdayError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(BirthdayError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_birthday(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_birthday_for_rng!(32);
impl_birthday_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn birthday32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut birthday = Birthday32::from_urng(&mut rng);
        let res = birthday.run(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, BirthdayVerdict::Pass);
    }

    #[test]
    fn birthday64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut birthday = Birthday64::from_urng(&mut rng);
        let res = birthday.run(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, BirthdayVerdict::Pass);
    }
}
