//! Serial test harness for RNGs: a multi-dimensional generalization of the
//! chi-squared test that checks for correlation between consecutive outputs.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for a serial (multi-dimensional uniformity) test.
///
/// Draws overlapping `dim`-tuples of consecutive `[0, 1)` samples and checks
/// whether they are uniformly distributed over the `dim`-dimensional unit cube.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SerialConfig {
    /// Number of `dim`-tuples drawn per test run.
    pub tuples: usize,
    /// Dimensionality of each tuple (2 = pairs, 3 = triples, ...).
    pub dim: usize,
    /// Number of equal-width bins per axis.
    pub bins_per_axis: usize,
    /// Maximum absolute z-score (`|z|`) permitted for a passing result.
    pub z_limit: f64,
}

impl Default for SerialConfig {
    /// Returns the default configuration: 1,000,000 pairs, dim 2, 16 bins per axis, z-limit 3.0.
    fn default() -> Self {
        Self {
            tuples: 1_000_000,
            dim: 2,
            bins_per_axis: 16,
            z_limit: 3.0,
        }
    }
}

impl SerialConfig {
    /// Validates the configuration, returning a [`SerialError`] describing the first problem found.
    fn validate(&self) -> Result<(), SerialError> {
        if self.tuples == 0 {
            return Err(SerialError::InvalidTuples {
                tuples: self.tuples,
            });
        }
        if self.dim < 2 {
            return Err(SerialError::InvalidDim { dim: self.dim });
        }
        if self.bins_per_axis < 2 {
            return Err(SerialError::InvalidBinsPerAxis {
                bins_per_axis: self.bins_per_axis,
            });
        }
        let total_bins = (self.bins_per_axis as u128).pow(self.dim as u32);
        if total_bins > (self.tuples as u128) {
            return Err(SerialError::TuplesLessThanBins {
                tuples: self.tuples,
                total_bins,
            });
        }
        if !self.z_limit.is_finite() || self.z_limit <= 0.0 {
            return Err(SerialError::InvalidZLimit {
                z_limit: self.z_limit,
            });
        }
        Ok(())
    }
}

/// Outcome of a single serial test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerialVerdict {
    /// The observed z-score was within the configured limit.
    Pass,
    /// The observed z-score exceeded the configured limit.
    Fail,
}

/// Full result of a single serial test run.
#[derive(Debug, Clone, PartialEq)]
pub struct SerialResult {
    /// Name of the test case.
    pub name: String,
    /// Number of tuples drawn.
    pub tuples: usize,
    /// Dimensionality of each tuple.
    pub dim: usize,
    /// Number of bins per axis.
    pub bins_per_axis: usize,
    /// Computed chi-squared statistic over the `dim`-dimensional histogram.
    pub chi2: f64,
    /// Degrees of freedom (`bins_per_axis^dim - 1`).
    pub df: f64,
    /// Normalized z-score: `(chi2 - df) / sqrt(2 * df)`.
    pub z_score: f64,
    /// z-score threshold used for the verdict.
    pub z_limit: f64,
    /// Final pass/fail verdict.
    pub verdict: SerialVerdict,
}

/// Errors that can occur while configuring or running a serial test.
#[derive(Debug, Error)]
pub enum SerialError {
    /// Zero tuples were requested.
    #[error("tuples must be greater than zero: tuples={tuples}")]
    InvalidTuples { tuples: usize },

    /// Fewer than two dimensions were requested.
    #[error("dim must be at least 2: dim={dim}")]
    InvalidDim { dim: usize },

    /// Fewer than two bins per axis were requested.
    #[error("bins_per_axis must be at least 2: bins_per_axis={bins_per_axis}")]
    InvalidBinsPerAxis { bins_per_axis: usize },

    /// Fewer tuples than histogram cells were requested (the test would be meaningless).
    #[error("tuples must be >= bins_per_axis^dim: tuples={tuples}, total_bins={total_bins}")]
    TuplesLessThanBins { tuples: usize, total_bins: u128 },

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
struct SerialCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

fn validate_case_name(name: String) -> Result<String, SerialError> {
    if name.trim().is_empty() {
        return Err(SerialError::EmptyCaseName);
    }
    Ok(name)
}

fn validate_sample(case: &str, sample_index: usize, x: f64) -> Result<(), SerialError> {
    if !x.is_finite() {
        return Err(SerialError::NonFiniteSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    if !(0.0..1.0).contains(&x) {
        return Err(SerialError::OutOfRangeSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    Ok(())
}

/// Runs a serial uniformity test for the given named sampler and configuration.
///
/// Draws `config.tuples` overlapping `config.dim`-tuples of consecutive samples,
/// bins them into a `config.dim`-dimensional histogram of `config.bins_per_axis`
/// cells per axis, computes the chi-squared statistic and its normalized z-score,
/// and returns a [`SerialResult`] carrying the [`SerialVerdict`].
fn run_serial(
    name: String,
    sampler: &mut dyn FnMut() -> f64,
    config: SerialConfig,
) -> Result<SerialResult, SerialError> {
    config.validate()?;

    let total_bins = config.bins_per_axis.pow(config.dim as u32);
    let mut counts = vec![0u64; total_bins];

    // Overlapping tuples: maintain a ring buffer of the last `dim` bin-indices.
    let mut window: Vec<usize> = Vec::with_capacity(config.dim);
    for sample_index in 0..(config.tuples + config.dim - 1) {
        let x = sampler();
        validate_sample(&name, sample_index, x)?;
        let bin = ((x * config.bins_per_axis as f64) as usize).min(config.bins_per_axis - 1);
        window.push(bin);
        if window.len() > config.dim {
            window.remove(0);
        }
        if window.len() == config.dim {
            let mut idx = 0usize;
            for &b in &window {
                idx = idx * config.bins_per_axis + b;
            }
            counts[idx] += 1;
        }
    }

    let expected = config.tuples as f64 / total_bins as f64;
    let chi2 = counts
        .iter()
        .map(|&c| {
            let delta = c as f64 - expected;
            (delta * delta) / expected
        })
        .sum::<f64>();
    let df = (total_bins - 1) as f64;
    let z_score = (chi2 - df) / (2.0 * df).sqrt();
    let verdict = if z_score.abs() <= config.z_limit {
        SerialVerdict::Pass
    } else {
        SerialVerdict::Fail
    };

    Ok(SerialResult {
        name,
        tuples: config.tuples,
        dim: config.dim,
        bins_per_axis: config.bins_per_axis,
        chi2,
        df,
        z_score,
        z_limit: config.z_limit,
        verdict,
    })
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_serial(name, &mut sampler, self.config)
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
                    let name = validate_case_name(name.into())?;
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
                    let name = validate_case_name(name.into())?;
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
                        let result = run_serial(case.name.clone(), case.sampler.as_mut(), self.config)?;
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
