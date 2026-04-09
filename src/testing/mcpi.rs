use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::{collections::HashSet, f64};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McPiConfig {
    pub pairs: usize,
    pub max_error_pct: f64,
}

impl Default for McPiConfig {
    fn default() -> Self {
        Self {
            pairs: 1_000_000,
            max_error_pct: 0.1,
        }
    }
}

impl McPiConfig {
    fn validate(&self) -> Result<(), McPiError> {
        if self.pairs == 0 {
            return Err(McPiError::InvalidPairs { pairs: self.pairs });
        }
        if !self.max_error_pct.is_finite() || self.max_error_pct <= 0.0 {
            return Err(McPiError::InvalidMaxErrorPct {
                max_error_pct: self.max_error_pct,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McPiVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq)]
pub struct McPiResult {
    pub name: String,
    pub pairs: usize,
    pub inside_circle: u64,
    pub pi_estimate: f64,
    pub absolute_error: f64,
    pub error_pct: f64,
    pub max_error_pct: f64,
    pub verdict: McPiVerdict,
}

#[derive(Debug, Error)]
pub enum McPiError {
    #[error("pairs must be greater than zero: pairs={pairs}")]
    InvalidPairs { pairs: usize },

    #[error("max_error_pct must be finite and > 0: max_error_pct={max_error_pct}")]
    InvalidMaxErrorPct { max_error_pct: f64 },

    #[error("test name must not be empty")]
    EmptyCaseName,

    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },

    #[error(
        "rng produced non-finite value: case={case}, sample_index={sample_index}, axis={axis}, value={value}"
    )]
    NonFiniteSample {
        case: String,
        sample_index: usize,
        axis: &'static str,
        value: f64,
    },

    #[error(
        "rng produced out-of-range value [0,1): case={case}, sample_index={sample_index}, axis={axis}, value={value}"
    )]
    OutOfRangeSample {
        case: String,
        sample_index: usize,
        axis: &'static str,
        value: f64,
    },
}

struct McPiCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

fn validate_case_name(name: String) -> Result<String, McPiError> {
    if name.trim().is_empty() {
        return Err(McPiError::EmptyCaseName);
    }
    Ok(name)
}

fn validate_sample(
    case: &str,
    sample_index: usize,
    axis: &'static str,
    x: f64,
) -> Result<(), McPiError> {
    if !x.is_finite() {
        return Err(McPiError::NonFiniteSample {
            case: case.to_string(),
            sample_index,
            axis,
            value: x,
        });
    }
    if !(0.0..1.0).contains(&x) {
        return Err(McPiError::OutOfRangeSample {
            case: case.to_string(),
            sample_index,
            axis,
            value: x,
        });
    }
    Ok(())
}

fn run_mcpi(
    name: String,
    sampler: &mut dyn FnMut() -> f64,
    config: McPiConfig,
) -> Result<McPiResult, McPiError> {
    config.validate()?;

    let mut inside_circle = 0u64;
    for sample_index in 0..config.pairs {
        let x = sampler();
        validate_sample(&name, sample_index, "x", x)?;
        let y = sampler();
        validate_sample(&name, sample_index, "y", y)?;
        if x * x + y * y < 1.0 {
            inside_circle += 1;
        }
    }

    let pi_estimate = 4.0 * inside_circle as f64 / config.pairs as f64;
    let absolute_error = (pi_estimate - f64::consts::PI).abs();
    let error_pct = (absolute_error / f64::consts::PI) * 100.0;
    let verdict = if error_pct <= config.max_error_pct {
        McPiVerdict::Pass
    } else {
        McPiVerdict::Fail
    };

    Ok(McPiResult {
        name,
        pairs: config.pairs,
        inside_circle,
        pi_estimate,
        absolute_error,
        error_pct,
        max_error_pct: config.max_error_pct,
        verdict,
    })
}

macro_rules! impl_mcpi_for_rng {
    ($bits:expr) => {
        paste::paste!{
            #[doc = concat!("Monte Carlo estimation of π using ", $bits, "-bit RNGs.")]
            pub struct [<McPi $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: McPiConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<McPi $bits>]<'a, R> {
                pub fn new(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: McPiConfig::default(),
                    }
                }

                pub fn with_config(rng: &'a mut R, config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                pub fn run(&mut self, name: impl Into<String>) -> Result<McPiResult, McPiError> {
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_mcpi(name, &mut sampler, self.config)
                }
            }

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
                pub fn new() -> Self {
                    Self::default()
                }

                pub fn with_config(config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
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
                ) -> Result<&mut Self, McPiError> {
                    let name = validate_case_name(name.into())?;
                    self.cases.push(McPiCase {
                        name,
                        sampler: Box::new(move || [<unit_f64_from_u $bits>](rng.nextu())),
                    });
                    Ok(self)
                }

                pub fn add_sampler<F>(
                    &mut self,
                    name: impl Into<String>,
                    sampler: F,
                ) -> Result<&mut Self, McPiError>
                where
                    F: FnMut() -> f64 + 'a,
                {
                    let name = validate_case_name(name.into())?;
                    self.cases.push(McPiCase {
                        name,
                        sampler: Box::new(sampler),
                    });
                    Ok(self)
                }

                pub fn run(&mut self) -> Result<Vec<McPiResult>, McPiError> {
                    let mut seen = HashSet::with_capacity(self.cases.len());
                    let mut out = Vec::with_capacity(self.cases.len());
                    for case in &mut self.cases {
                        if !seen.insert(case.name.clone()) {
                            return Err(McPiError::DuplicateCaseName {
                                name: case.name.clone(),
                            });
                        }
                        let result = run_mcpi(case.name.clone(), case.sampler.as_mut(), self.config)?;
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
    use crate::prelude::*;

    #[test]
    fn mcpi32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut mcpi = McPi32::new(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt19937)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }

    #[test]
    fn mcpi64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut mcpi = McPi64::new(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt1993764)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }
}
