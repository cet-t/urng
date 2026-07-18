//! Runs test harness for RNGs: detects sequential correlation by analyzing
//! the lengths of monotonic ascending/descending runs in the output sequence.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for a runs test.
///
/// Draws `samples` values and counts the number of maximal monotonic runs
/// (up or down). Under the null hypothesis of independence, the expected
/// number of runs and its variance are known in closed form (Knuth TAOCP
/// Vol. 2, 3.3.2), which is used to compute a z-score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunsConfig {
    /// Number of random samples drawn per test run.
    pub samples: usize,
    /// Maximum absolute z-score (`|z|`) permitted for a passing result.
    pub z_limit: f64,
}

impl Default for RunsConfig {
    /// Returns the default configuration: 1,000,000 samples, z-limit 3.0.
    fn default() -> Self {
        Self {
            samples: 1_000_000,
            z_limit: 3.0,
        }
    }
}

impl RunsConfig {
    /// Validates the configuration, returning a [`RunsError`] describing the first problem found.
    fn validate(&self) -> Result<(), RunsError> {
        if self.samples < 2 {
            return Err(RunsError::InvalidSamples {
                samples: self.samples,
            });
        }
        if !self.z_limit.is_finite() || self.z_limit <= 0.0 {
            return Err(RunsError::InvalidZLimit {
                z_limit: self.z_limit,
            });
        }
        Ok(())
    }
}

/// Outcome of a single runs test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunsVerdict {
    /// The observed z-score was within the configured limit.
    Pass,
    /// The observed z-score exceeded the configured limit.
    Fail,
}

/// Full result of a single runs test run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunsResult {
    /// Name of the test case.
    pub name: String,
    /// Number of samples drawn.
    pub samples: usize,
    /// Observed number of monotonic runs.
    pub runs: usize,
    /// Expected number of runs under independence: `(2n - 1) / 3`.
    pub expected_runs: f64,
    /// Normalized z-score of the observed run count.
    pub z_score: f64,
    /// z-score threshold used for the verdict.
    pub z_limit: f64,
    /// Final pass/fail verdict.
    pub verdict: RunsVerdict,
}

/// Errors that can occur while configuring or running a runs test.
#[derive(Debug, Error)]
pub enum RunsError {
    /// Fewer than two samples were requested.
    #[error("samples must be at least 2: samples={samples}")]
    InvalidSamples { samples: usize },

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

struct RunsCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

fn validate_case_name(name: String) -> Result<String, RunsError> {
    if name.trim().is_empty() {
        return Err(RunsError::EmptyCaseName);
    }
    Ok(name)
}

fn validate_sample(case: &str, sample_index: usize, x: f64) -> Result<(), RunsError> {
    if !x.is_finite() {
        return Err(RunsError::NonFiniteSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    if !(0.0..1.0).contains(&x) {
        return Err(RunsError::OutOfRangeSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    Ok(())
}

/// Runs a runs test (Wald-Wolfowitz-style, up/down variant) for the given
/// named sampler and configuration.
///
/// Counts the number of maximal monotonic ascending/descending runs in the
/// sequence and compares it to the closed-form expected value and variance
/// under independence (Knuth TAOCP Vol. 2, 3.3.2, eq. (1)-(2)):
/// `E[runs] = (2n - 1) / 3`, `Var[runs] = (16n - 29) / 90`.
fn run_runs(
    name: String,
    sampler: &mut dyn FnMut() -> f64,
    config: RunsConfig,
) -> Result<RunsResult, RunsError> {
    config.validate()?;

    let n = config.samples;
    let mut prev = sampler();
    validate_sample(&name, 0, prev)?;

    let mut runs = 1usize;
    let mut ascending: Option<bool> = None;

    for sample_index in 1..n {
        let x = sampler();
        validate_sample(&name, sample_index, x)?;
        let is_ascending = x > prev;
        match ascending {
            Some(prev_dir) if prev_dir == is_ascending => {}
            _ => runs += 1,
        }
        ascending = Some(is_ascending);
        prev = x;
    }

    let n_f = n as f64;
    let expected_runs = (2.0 * n_f - 1.0) / 3.0;
    let variance = (16.0 * n_f - 29.0) / 90.0;
    let z_score = (runs as f64 - expected_runs) / variance.sqrt();
    let verdict = if z_score.abs() <= config.z_limit {
        RunsVerdict::Pass
    } else {
        RunsVerdict::Fail
    };

    Ok(RunsResult {
        name,
        samples: n,
        runs,
        expected_runs,
        z_score,
        z_limit: config.z_limit,
        verdict,
    })
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_runs(name, &mut sampler, self.config)
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
                    let name = validate_case_name(name.into())?;
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
                    let name = validate_case_name(name.into())?;
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
                        let result = run_runs(case.name.clone(), case.sampler.as_mut(), self.config)?;
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
