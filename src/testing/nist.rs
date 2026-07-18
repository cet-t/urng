//! NIST SP 800-22 (rev. 1a) statistical test suite, bit-level subset.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine. See
//! [`cribler::nist`] for the sub-test list and math.

use crate::rng::{Rng32, Rng64};
use std::collections::HashSet;

pub use cribler::{NistConfig, NistError, NistItemResult, NistResult, NistTest, NistVerdict};

struct NistCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> u64 + 'a>,
    word_bits: u32,
}

macro_rules! impl_nist_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("NIST SP 800-22 bit-level test suite for `Rng", $bits, "` generators.")]
            pub struct [<Nist $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: NistConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<Nist $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: NistConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: NistConfig) -> Result<Self, NistError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> NistConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: NistConfig) -> Result<(), NistError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<NistResult, NistError> {
                    let mut sampler = || self.rng.nextu() as u64;
                    cribler::run_nist(name, &mut sampler, $bits, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<Nist $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `Nist", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: NistConfig,
                ) -> Result<Self, NistError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<NistSuite $bits>]<'a> {
                config: NistConfig,
                cases: Vec<NistCase<'a>>,
            }

            impl<'a> Default for [<NistSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: NistConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<NistSuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: NistConfig) -> Result<Self, NistError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> NistConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: NistConfig) -> Result<(), NistError> {
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
                ) -> Result<&mut Self, NistError> {
                    let name = name.into();
                    self.cases.push(NistCase {
                        name,
                        sampler: Box::new(move || rng.nextu() as u64),
                        word_bits: $bits,
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<NistResult>, NistError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(NistError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_nist(
                            case.name.clone(),
                            case.sampler.as_mut(),
                            case.word_bits,
                            self.config,
                        )?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_nist_for_rng!(32);
impl_nist_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn nist32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut nist = Nist32::from_urng(&mut rng);
        let res = nist.run(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, NistVerdict::Pass);
    }

    #[test]
    fn nist64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut nist = Nist64::from_urng(&mut rng);
        let res = nist.run(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, NistVerdict::Pass);
    }
}
