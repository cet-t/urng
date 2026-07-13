//! Monte Carlo π-estimation test harness for RNGs.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::{collections::HashSet, f64};
use thiserror::Error;

/// Configuration for a Monte Carlo π-estimation test.
///
/// Controls how many random point pairs are sampled and how large a relative
/// error from π is still considered a passing result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McPiConfig {
    /// Number of random `(x, y)` point pairs drawn per test run.
    pub pairs: usize,
    /// Maximum allowed relative error (in percent) from the true value of π.
    pub max_error_pct: f64,
}

impl Default for McPiConfig {
    /// Returns the default configuration: 1,000,000 point pairs, max error 0.1%.
    fn default() -> Self {
        Self {
            pairs: 1_000_000,
            max_error_pct: 0.1,
        }
    }
}

impl McPiConfig {
    /// Validates the configuration, returning a [`McPiError`] describing the first problem found.
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

/// Outcome of a single Monte Carlo π-estimation test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McPiVerdict {
    /// The estimated π was within `max_error_pct` of the true value.
    Pass,
    /// The estimated π deviated too far from the true value.
    Fail,
}

/// Full result of a single Monte Carlo π-estimation test run.
#[derive(Debug, Clone, PartialEq)]
pub struct McPiResult {
    /// Name of the test case.
    pub name: String,
    /// Number of point pairs sampled.
    pub pairs: usize,
    /// Number of points that fell inside the unit quarter-circle.
    pub inside_circle: u64,
    /// Estimated value of π (`4 * inside / pairs`).
    pub pi_estimate: f64,
    /// Absolute error `|estimate - π|`.
    pub absolute_error: f64,
    /// Relative error as a percentage of π.
    pub error_pct: f64,
    /// Relative-error threshold used for the verdict.
    pub max_error_pct: f64,
    /// Final pass/fail verdict.
    pub verdict: McPiVerdict,
}

/// Errors that can occur while configuring or running a Monte Carlo π test.
#[derive(Debug, Error)]
pub enum McPiError {
    /// Zero point pairs were requested.
    #[error("pairs must be greater than zero: pairs={pairs}")]
    InvalidPairs { pairs: usize },

    /// A non-positive or non-finite `max_error_pct` was supplied.
    #[error("max_error_pct must be finite and > 0: max_error_pct={max_error_pct}")]
    InvalidMaxErrorPct { max_error_pct: f64 },

    /// A test case name was empty or whitespace-only.
    #[error("test name must not be empty")]
    EmptyCaseName,

    /// Two cases in the same suite shared a name.
    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },

    /// The generator produced a non-finite (`NaN`/`inf`) value.
    #[error(
        "rng produced non-finite value: case={case}, sample_index={sample_index}, axis={axis}, value={value}"
    )]
    NonFiniteSample {
        case: String,
        sample_index: usize,
        axis: &'static str,
        value: f64,
    },

    /// The generator produced a value outside `[0, 1)`.
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

/// A single named test case: a name plus a closure that produces `[0, 1)` floats.
struct McPiCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

/// Validates a test-case name, rejecting empty or whitespace-only names.
fn validate_case_name(name: String) -> Result<String, McPiError> {
    if name.trim().is_empty() {
        return Err(McPiError::EmptyCaseName);
    }
    Ok(name)
}

/// Checks that a sample is finite and lies in `[0, 1)`, returning a descriptive error otherwise.
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

/// Runs a Monte Carlo π-estimation test for the given named sampler and configuration.
///
/// Samples `config.pairs` points uniformly in `[0, 1)²`, counts how many fall inside the
/// unit quarter-circle, estimates π as `4 * inside / pairs`, and returns a [`McPiResult`]
/// carrying the [`McPiVerdict`].
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
            #[doc = ""]
            #[doc = "Wraps a mutable reference to a generator and counts how many of its `[0, 1)`"]
            #[doc = "point pairs fall inside the unit quarter-circle to estimate π."]
            pub struct [<McPi $bits>]<'a, R: [<Rng $bits>] + 'a> {
                rng: &'a mut R,
                config: McPiConfig,
            }

            impl<'a, R: [<Rng $bits>] + 'a> [<McPi $bits>]<'a, R> {
                #[doc = concat!("Creates a tester for `", stringify!([<Rng $bits>]), "` using the default configuration.")]
                pub fn from_urng(rng: &'a mut R) -> Self {
                    Self {
                        rng,
                        config: McPiConfig::default(),
                    }
                }

                #[deprecated(note = "use `from_urng` instead")]
                #[doc = "Deprecated alias for [`from_urng`](Self::from_urng)."]
                pub fn new(rng: &'a mut R) -> Self {
                    Self::from_urng(rng)
                }

                #[doc = "Creates a tester with an explicit, validated configuration."]
                pub fn with_config(rng: &'a mut R, config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self { rng, config })
                }

                #[doc = "Returns the current configuration."]
                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                #[doc = "Replaces the configuration after validating it."]
                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                #[doc = "Runs the Monte Carlo test, returning a [`McPiResult`] (or the first validation error)."]
                pub fn run(&mut self, name: impl Into<String>) -> Result<McPiResult, McPiError> {
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_mcpi(name, &mut sampler, self.config)
                }
            }

            #[cfg(feature = "rand")]
            impl<'a, R: rand_core::Rng + 'a> [<McPi $bits>]<'a, crate::testing::rand_adapter::RandAdapter<R>> {
                #[doc = concat!("Creates a new `McPi", $bits, "` from any `rand_core::Rng` implementation, without manually wrapping it in `RandAdapter`.")]
                pub fn from_rand(rng: &'a mut R) -> Self {
                    Self::from_urng(crate::testing::rand_adapter::RandAdapter::from_mut(rng))
                }

                #[doc = "Creates a tester with an explicit configuration from any `rand_core::Rng`."]
                pub fn with_config_from_rand(
                    rng: &'a mut R,
                    config: McPiConfig,
                ) -> Result<Self, McPiError> {
                    Self::with_config(crate::testing::rand_adapter::RandAdapter::from_mut(rng), config)
                }
            }

            #[doc = concat!("A suite that runs multiple Monte Carlo π test cases and collects their [`McPiResult`]s.")]
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
                #[doc = "Creates an empty suite with the default configuration."]
                pub fn new() -> Self {
                    Self::default()
                }

                #[doc = "Creates an empty suite with an explicit, validated configuration."]
                pub fn with_config(config: McPiConfig) -> Result<Self, McPiError> {
                    config.validate()?;
                    Ok(Self {
                        config,
                        cases: Vec::new(),
                    })
                }

                #[doc = "Returns the current configuration."]
                pub fn config(&self) -> McPiConfig {
                    self.config
                }

                #[doc = "Replaces the configuration after validating it."]
                pub fn set_config(&mut self, config: McPiConfig) -> Result<(), McPiError> {
                    config.validate()?;
                    self.config = config;
                    Ok(())
                }

                #[doc = "Returns the number of registered cases."]
                pub fn len(&self) -> usize {
                    self.cases.len()
                }

                #[doc = "Returns `true` if no cases are registered."]
                pub fn is_empty(&self) -> bool {
                    self.cases.is_empty()
                }

                #[doc = "Removes all registered cases."]
                pub fn clear(&mut self) {
                    self.cases.clear();
                }

                #[doc = concat!("Registers a `", stringify!([<Rng $bits>]), "` generator as a named test case.")]
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

                #[doc = "Registers an arbitrary `[0, 1)` sampler closure as a named test case."]
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

                #[doc = "Runs every case, returning a [`McPiResult`] per case (or the first error)."]
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
    use crate::Sfmt19937;
    use crate::Sfmt1993764;

    #[test]
    fn mcpi32_works() {
        let mut rng = Sfmt19937::new(0);
        let mut mcpi = McPi32::from_urng(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt19937)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }

    #[test]
    fn mcpi64_works() {
        let mut rng = Sfmt1993764::new(0);
        let mut mcpi = McPi64::from_urng(&mut rng);
        let res = unsafe { mcpi.run(stringify!(Sfmt1993764)).unwrap_unchecked() };
        assert_eq!(res.verdict, McPiVerdict::Pass);
    }
}
