//! Statistical test harness for RNGs based on the chi-squared test.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSqConfig {
    pub samples: usize,
    pub bins: usize,
    pub z_limit: f64,
}

impl Default for ChiSqConfig {
    fn default() -> Self {
        Self {
            samples: 1_000_000,
            bins: 256,
            z_limit: 3.0,
        }
    }
}

impl ChiSqConfig {
    fn validate(&self) -> Result<(), ChiSqError> {
        if self.samples == 0 {
            return Err(ChiSqError::InvalidSamples {
                samples: self.samples,
            });
        }
        if self.bins < 2 {
            return Err(ChiSqError::InvalidBins { bins: self.bins });
        }
        if self.samples < self.bins {
            return Err(ChiSqError::SamplesLessThanBins {
                samples: self.samples,
                bins: self.bins,
            });
        }
        if !self.z_limit.is_finite() || self.z_limit <= 0.0 {
            return Err(ChiSqError::InvalidZLimit {
                z_limit: self.z_limit,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChiSqVerdict {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChiSqResult {
    pub name: String,
    pub samples: usize,
    pub bins: usize,
    pub chi2: f64,
    pub df: f64,
    pub z_score: f64,
    pub z_limit: f64,
    pub verdict: ChiSqVerdict,
}

#[derive(Debug, Error)]
pub enum ChiSqError {
    #[error("samples must be greater than zero: samples={samples}")]
    InvalidSamples { samples: usize },

    #[error("bins must be at least 2: bins={bins}")]
    InvalidBins { bins: usize },

    #[error("samples must be >= bins: samples={samples}, bins={bins}")]
    SamplesLessThanBins { samples: usize, bins: usize },

    #[error("z_limit must be finite and > 0: z_limit={z_limit}")]
    InvalidZLimit { z_limit: f64 },

    #[error("test name must not be empty")]
    EmptyCaseName,

    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },

    #[error(
        "rng produced non-finite value: case={case}, sample_index={sample_index}, value={value}"
    )]
    NonFiniteSample {
        case: String,
        sample_index: usize,
        value: f64,
    },

    #[error(
        "rng produced out-of-range value [0,1): case={case}, sample_index={sample_index}, value={value}"
    )]
    OutOfRangeSample {
        case: String,
        sample_index: usize,
        value: f64,
    },
}

struct ChiSqCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

fn validate_case_name(name: String) -> Result<String, ChiSqError> {
    if name.trim().is_empty() {
        return Err(ChiSqError::EmptyCaseName);
    }
    Ok(name)
}

fn validate_sample(case: &str, sample_index: usize, x: f64) -> Result<(), ChiSqError> {
    if !x.is_finite() {
        return Err(ChiSqError::NonFiniteSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    if !(0.0..1.0).contains(&x) {
        return Err(ChiSqError::OutOfRangeSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    Ok(())
}

fn run_chisq(
    name: String,
    sampler: &mut dyn FnMut() -> f64,
    config: ChiSqConfig,
) -> Result<ChiSqResult, ChiSqError> {
    config.validate()?;
    let mut counts = vec![0u64; config.bins];

    for sample_index in 0..config.samples {
        let x = sampler();
        validate_sample(&name, sample_index, x)?;
        let bin = ((x * config.bins as f64) as usize).min(config.bins - 1);
        counts[bin] += 1;
    }

    let expected = config.samples as f64 / config.bins as f64;
    let chi2 = counts
        .iter()
        .map(|&c| {
            let delta = c as f64 - expected;
            (delta * delta) / expected
        })
        .sum::<f64>();
    let df = (config.bins - 1) as f64;
    let z_score = (chi2 - df) / (2.0 * df).sqrt();
    let verdict = if z_score.abs() <= config.z_limit {
        ChiSqVerdict::Pass
    } else {
        ChiSqVerdict::Fail
    };

    Ok(ChiSqResult {
        name,
        samples: config.samples,
        bins: config.bins,
        chi2,
        df,
        z_score,
        z_limit: config.z_limit,
        verdict,
    })
}

// impl_chisq_for_rng!(ChiSq32, ChiSqSuite32, Rng32, unit_f64_from_u32);
macro_rules! impl_chisq_for_rng {
    ($bits:expr) => {
        paste::paste! {
            pub struct [<ChiSq $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: ChiSqConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<ChiSq $bits>]<'a, R> {
                pub fn new(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: ChiSqConfig::default(),
                    }
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_chisq(name, &mut sampler, self.config)
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
                    let name = validate_case_name(name.into())?;
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
                    let name = validate_case_name(name.into())?;
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
                        let result = run_chisq(case.name.clone(), case.sampler.as_mut(), self.config)?;
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
    use crate::prelude::*;

    #[test]
    fn chisq32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut chisq = ChiSq32::new(&mut rng);
        let res = unsafe { chisq.run(stringify!(Sfmt19937)).unwrap_unchecked() };
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }

    #[test]
    fn chisq64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut chisq = ChiSq64::new(&mut rng);
        let res = unsafe { chisq.run(stringify!(Sfmt1993764)).unwrap_unchecked() };
        assert_eq!(res.verdict, ChiSqVerdict::Pass);
    }
}
