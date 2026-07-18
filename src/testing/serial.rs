//! Serial test harness for RNGs: a multi-dimensional generalization of the
//! chi-squared test that checks for correlation between consecutive outputs.
//!
//! Thin `Rng32`/`Rng64`-typed wrappers over the [`cribler`] engine.

use crate::rng::{Rng32, Rng64};
use cribler::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;

pub use cribler::{SerialConfig, SerialError, SerialResult, SerialVerdict};

struct SerialCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

macro_rules! impl_serial_for_rng {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("Serial (multi-dimensional uniformity) tester for `Rng", $bits, "` generators.")]
            pub struct [<Serial $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: SerialConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<Serial $bits>]<'a, R> {
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: SerialConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: SerialConfig) -> Result<Self, SerialError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> SerialConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: SerialConfig) -> Result<(), SerialError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<SerialResult, SerialError> {
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    cribler::run_serial(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<Serial $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `Serial", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: SerialConfig,
                ) -> Result<Self, SerialError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            pub struct [<SerialSuite $bits>]<'a> {
                config: SerialConfig,
                cases: Vec<SerialCase<'a>>,
            }

            impl<'a> Default for [<SerialSuite $bits>]<'a> {
                fn default() -> Self {
                    Self {
                        config: SerialConfig::default(),
                        cases: Vec::new(),
                    }
                }
            }

            impl<'a> [<SerialSuite $bits>]<'a> {
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: SerialConfig) -> Result<Self, SerialError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> SerialConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: SerialConfig) -> Result<(), SerialError> {
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
                ) -> Result<&mut Self, SerialError> {
                    let name = name.into();
                    self.cases.push(SerialCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, SerialError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = name.into();
                    self.cases.push(SerialCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<SerialResult>, SerialError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(SerialError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = cribler::run_serial(case.name.clone(), case.sampler.as_mut(), self.config)?;
                        out.push(result);
                    }
                    Ok(out)
                }
            }
        }
    };
}

impl_serial_for_rng!(32);
impl_serial_for_rng!(64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn serial32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut serial = Serial32::from_urng(&mut rng);
        let res = serial.run(stringify!(Sfmt19937)).unwrap();
        assert_eq!(res.verdict, SerialVerdict::Pass);
    }

    #[test]
    fn serial64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut serial = Serial64::from_urng(&mut rng);
        let res = serial.run(stringify!(Sfmt1993764)).unwrap();
        assert_eq!(res.verdict, SerialVerdict::Pass);
    }
}
