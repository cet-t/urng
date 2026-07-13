//! Statistical test harness for RNGs based on the chi-squared test.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for a chi-squared randomness test.
///
/// Controls how many samples are drawn, how the `[0, 1)` range is bucketed, and
/// how large a deviation (in normalized z-score units) is still considered passing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSqConfig {
    /// Number of random samples drawn per test run.
    pub samples: usize,
    /// Number of equal-width histogram bins over `[0, 1)`.
    pub bins: usize,
    /// Maximum absolute z-score (`|z|`) permitted for a passing result.
    pub z_limit: f64,
}

impl Default for ChiSqConfig {
    /// Returns the default configuration: 1,000,000 samples, 256 bins, z-limit 3.0.
    fn default() -> Self {
        Self {
            samples: 1_000_000,
            bins: 256,
            z_limit: 3.0,
        }
    }
}

impl ChiSqConfig {
    /// Validates the configuration, returning a [`ChiSqError`] describing the first problem found.
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

/// Outcome of a single chi-squared test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChiSqVerdict {
    /// The observed z-score was within the configured limit.
    Pass,
    /// The observed z-score exceeded the configured limit.
    Fail,
}

/// Full result of a single chi-squared test run.
#[derive(Debug, Clone, PartialEq)]
pub struct ChiSqResult {
    /// Name of the test case.
    pub name: String,
    /// Number of samples drawn.
    pub samples: usize,
    /// Number of histogram bins.
    pub bins: usize,
    /// Computed chi-squared statistic.
    pub chi2: f64,
    /// Degrees of freedom (`bins - 1`).
    pub df: f64,
    /// Normalized z-score: `(chi2 - df) / sqrt(2 * df)`.
    pub z_score: f64,
    /// z-score threshold used for the verdict.
    pub z_limit: f64,
    /// Final pass/fail verdict.
    pub verdict: ChiSqVerdict,
}

/// Errors that can occur while configuring or running a chi-squared test.
#[derive(Debug, Error)]
pub enum ChiSqError {
    /// Zero samples were requested.
    #[error("samples must be greater than zero: samples={samples}")]
    InvalidSamples { samples: usize },

    /// Fewer than two bins were requested.
    #[error("bins must be at least 2: bins={bins}")]
    InvalidBins { bins: usize },

    /// Fewer samples than bins were requested (the test would be meaningless).
    #[error("samples must be >= bins: samples={samples}, bins={bins}")]
    SamplesLessThanBins { samples: usize, bins: usize },

    /// A non-positive or non-finite z-limit was supplied.
    #[error("z_limit must be finite and > 0: z_limit={z_limit}")]
    InvalidZLimit { z_limit: f64 },

    /// A test case name was empty or whitespace-only.
    #[error("test name must not be empty")]
    EmptyCaseName,

    /// Two cases in the same suite shared a name.
    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },

    /// The generator produced a non-finite (`NaN`/`inf`) value.
    #[error(
        "rng produced non-finite value: case={case}, sample_index={sample_index}, value={value}"
    )]
    NonFiniteSample {
        case: String,
        sample_index: usize,
        value: f64,
    },

    /// The generator produced a value outside `[0, 1)`.
    #[error(
        "rng produced out-of-range value [0,1): case={case}, sample_index={sample_index}, value={value}"
    )]
    OutOfRangeSample {
        case: String,
        sample_index: usize,
        value: f64,
    },
}

/// A single named test case: a name plus a closure that produces `[0, 1)` floats.
struct ChiSqCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

/// Validates a test-case name, rejecting empty or whitespace-only names.
fn validate_case_name(name: String) -> Result<String, ChiSqError> {
    if name.trim().is_empty() {
        return Err(ChiSqError::EmptyCaseName);
    }
    Ok(name)
}

/// Checks that a sample is finite and lies in `[0, 1)`, returning a descriptive error otherwise.
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

/// Runs a chi-squared uniformity test for the given named sampler and configuration.
///
/// Draws `config.samples` values, bins them into `config.bins` equal-width buckets over
/// `[0, 1)`, computes the chi-squared statistic and its normalized z-score, and returns a
/// [`ChiSqResult`] carrying the [`ChiSqVerdict`].
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_chisq(name, &mut sampler, self.config)
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
