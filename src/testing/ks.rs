//! Kolmogorov-Smirnov test harness for RNGs: checks the maximum deviation
//! between the empirical CDF of the samples and the ideal uniform CDF.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{unit_f64_from_u32, unit_f64_from_u64};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for a Kolmogorov-Smirnov uniformity test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsConfig {
    /// Number of random samples drawn per test run.
    pub samples: usize,
    /// Maximum allowed KS statistic `D`, expressed via the asymptotic
    /// significance level `alpha` (e.g. `0.01` for a 1% false-positive rate).
    pub alpha: f64,
}

impl Default for KsConfig {
    /// Returns the default configuration: 1,000,000 samples, alpha 0.01.
    fn default() -> Self {
        Self {
            samples: 1_000_000,
            alpha: 0.01,
        }
    }
}

impl KsConfig {
    /// Validates the configuration, returning a [`KsError`] describing the first problem found.
    fn validate(&self) -> Result<(), KsError> {
        if self.samples == 0 {
            return Err(KsError::InvalidSamples {
                samples: self.samples,
            });
        }
        if !self.alpha.is_finite() || !(0.0..1.0).contains(&self.alpha) {
            return Err(KsError::InvalidAlpha { alpha: self.alpha });
        }
        Ok(())
    }

    /// Critical value for the two-sided one-sample KS statistic, using the
    /// standard asymptotic (Kolmogorov) approximation:
    /// `D_alpha = sqrt(-0.5 * ln(alpha / 2)) / sqrt(n)`.
    fn critical_value(&self) -> f64 {
        (-0.5 * (self.alpha / 2.0).ln()).sqrt() / (self.samples as f64).sqrt()
    }
}

/// Outcome of a single Kolmogorov-Smirnov test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsVerdict {
    /// The observed KS statistic was within the critical value.
    Pass,
    /// The observed KS statistic exceeded the critical value.
    Fail,
}

/// Full result of a single Kolmogorov-Smirnov test run.
#[derive(Debug, Clone, PartialEq)]
pub struct KsResult {
    /// Name of the test case.
    pub name: String,
    /// Number of samples drawn.
    pub samples: usize,
    /// Observed KS statistic: `max |F_empirical(x) - x|`.
    pub d_statistic: f64,
    /// Critical value at the configured significance level.
    pub critical_value: f64,
    /// Significance level used for the verdict.
    pub alpha: f64,
    /// Final pass/fail verdict.
    pub verdict: KsVerdict,
}

/// Errors that can occur while configuring or running a KS test.
#[derive(Debug, Error)]
pub enum KsError {
    /// Zero samples were requested.
    #[error("samples must be greater than zero: samples={samples}")]
    InvalidSamples { samples: usize },

    /// An out-of-range or non-finite significance level was supplied.
    #[error("alpha must be finite and in (0, 1): alpha={alpha}")]
    InvalidAlpha { alpha: f64 },

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

struct KsCase<'a> {
    name: String,
    sampler: Box<dyn FnMut() -> f64 + 'a>,
}

fn validate_case_name(name: String) -> Result<String, KsError> {
    if name.trim().is_empty() {
        return Err(KsError::EmptyCaseName);
    }
    Ok(name)
}

fn validate_sample(case: &str, sample_index: usize, x: f64) -> Result<(), KsError> {
    if !x.is_finite() {
        return Err(KsError::NonFiniteSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    if !(0.0..1.0).contains(&x) {
        return Err(KsError::OutOfRangeSample {
            case: case.to_string(),
            sample_index,
            value: x,
        });
    }
    Ok(())
}

/// Runs a one-sample Kolmogorov-Smirnov uniformity test for the given named
/// sampler and configuration.
///
/// Draws `config.samples` values, sorts them, and computes the maximum
/// deviation between the empirical CDF and the ideal `Uniform(0, 1)` CDF:
/// `D = max_i max(i/n - x_(i), x_(i) - (i-1)/n)`.
fn run_ks(
    name: String,
    sampler: &mut dyn FnMut() -> f64,
    config: KsConfig,
) -> Result<KsResult, KsError> {
    config.validate()?;

    let n = config.samples;
    let mut xs = Vec::with_capacity(n);
    for sample_index in 0..n {
        let x = sampler();
        validate_sample(&name, sample_index, x)?;
        xs.push(x);
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n_f = n as f64;
    let mut d_statistic = 0.0f64;
    for (i, &x) in xs.iter().enumerate() {
        let i_f = (i + 1) as f64;
        let d_plus = i_f / n_f - x;
        let d_minus = x - (i_f - 1.0) / n_f;
        d_statistic = d_statistic.max(d_plus).max(d_minus);
    }

    let critical_value = config.critical_value();
    let verdict = if d_statistic <= critical_value {
        KsVerdict::Pass
    } else {
        KsVerdict::Fail
    };

    Ok(KsResult {
        name,
        samples: n,
        d_statistic,
        critical_value,
        alpha: config.alpha,
        verdict,
    })
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || [<unit_f64_from_u $bits>](self.rng.nextu());
                    run_ks(name, &mut sampler, self.config)
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
                    let name = validate_case_name(name.into())?;
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
                    let name = validate_case_name(name.into())?;
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
                        let result = run_ks(case.name.clone(), case.sampler.as_mut(), self.config)?;
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
