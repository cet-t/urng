//! "Paranoid" meta-test harness: elevates any single statistical test into a
//! battery-level test, following the "Testing Strategy and Result
//! Interpretation" guidance of NIST SP 800-22 rev. 1a, §4.2.
//!
//! A single p-value can pass by chance even for a bad generator, and can
//! fail by chance even for a good one. §4.2 recommends running the test many
//! times over independent samples and checking two things about the
//! resulting collection of p-values:
//!
//! 1. **Proportion of passes** (§4.2.1): the fraction of trials with
//!    `p_value >= trial_alpha` should fall within `p_hat +/- 3*sqrt(p_hat*(1-p_hat)/trials)`,
//!    where `p_hat = 1 - trial_alpha`.
//! 2. **Uniformity of p-values** (§4.2.2): under the null hypothesis the
//!    p-values themselves should be uniformly distributed over `[0, 1)`.
//!    This is checked with a Kolmogorov-Smirnov test against `Uniform(0, 1)`.
//!
//! This module is generic over *how* a p-value is produced per trial: wrap a
//! call to [`crate::testing::ChiSq32`], [`crate::testing::Nist32`], or any
//! other test in a closure that returns a p-value, and [`run_paranoid`] (or
//! [`ParanoidSuite32`]/[`ParanoidSuite64`] for batching named cases) applies
//! the meta-level checks.

use std::collections::HashSet;
use thiserror::Error;

use crate::testing::_internal::erfc;

/// Configuration for a paranoid (battery-level) meta-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParanoidConfig {
    /// Number of independent trials to run.
    pub trials: usize,
    /// Per-trial significance level: a trial "passes" when `p_value >= trial_alpha`.
    pub trial_alpha: f64,
    /// Significance level for the Kolmogorov-Smirnov uniformity check over the
    /// collected p-values (NIST recommends a small value, e.g. `0.0001`, to
    /// avoid an excessive false-positive rate at the meta-test level).
    pub uniformity_alpha: f64,
}

impl Default for ParanoidConfig {
    /// Returns the default configuration: 100 trials, trial alpha 0.01, uniformity alpha 0.0001.
    fn default() -> Self {
        Self {
            trials: 100,
            trial_alpha: 0.01,
            uniformity_alpha: 0.0001,
        }
    }
}

impl ParanoidConfig {
    /// Validates the configuration, returning a [`ParanoidError`] describing the first problem found.
    fn validate(&self) -> Result<(), ParanoidError> {
        if self.trials < 2 {
            return Err(ParanoidError::InvalidTrials {
                trials: self.trials,
            });
        }
        if !self.trial_alpha.is_finite() || !(0.0..1.0).contains(&self.trial_alpha) {
            return Err(ParanoidError::InvalidTrialAlpha {
                trial_alpha: self.trial_alpha,
            });
        }
        if !self.uniformity_alpha.is_finite() || !(0.0..1.0).contains(&self.uniformity_alpha) {
            return Err(ParanoidError::InvalidUniformityAlpha {
                uniformity_alpha: self.uniformity_alpha,
            });
        }
        Ok(())
    }
}

/// Outcome of a single check within the paranoid meta-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParanoidVerdict {
    /// The check was within tolerance.
    Pass,
    /// The check exceeded tolerance.
    Fail,
}

/// Full result of a paranoid meta-test run.
#[derive(Debug, Clone, PartialEq)]
pub struct ParanoidResult {
    /// Name of the test case.
    pub name: String,
    /// Number of trials run.
    pub trials: usize,
    /// Number of trials with `p_value >= trial_alpha`.
    pub passes: usize,
    /// Observed pass proportion (`passes / trials`).
    pub pass_proportion: f64,
    /// Lower bound of the acceptable pass-proportion range (NIST §4.2.1).
    pub proportion_lower: f64,
    /// Upper bound of the acceptable pass-proportion range (NIST §4.2.1).
    pub proportion_upper: f64,
    /// Verdict of the pass-proportion check.
    pub proportion_verdict: ParanoidVerdict,
    /// Kolmogorov-Smirnov `D` statistic of the p-value distribution against `Uniform(0, 1)`.
    pub ks_d_statistic: f64,
    /// Critical value for the KS uniformity check at `uniformity_alpha`.
    pub ks_critical_value: f64,
    /// Verdict of the p-value uniformity check.
    pub uniformity_verdict: ParanoidVerdict,
    /// Overall verdict: [`ParanoidVerdict::Pass`] iff both sub-checks passed.
    pub verdict: ParanoidVerdict,
}

/// Errors that can occur while configuring or running a paranoid meta-test.
#[derive(Debug, Error)]
pub enum ParanoidError {
    /// Fewer than two trials were requested.
    #[error("trials must be at least 2: trials={trials}")]
    InvalidTrials { trials: usize },

    /// An out-of-range or non-finite trial significance level was supplied.
    #[error("trial_alpha must be finite and in (0, 1): trial_alpha={trial_alpha}")]
    InvalidTrialAlpha { trial_alpha: f64 },

    /// An out-of-range or non-finite uniformity significance level was supplied.
    #[error("uniformity_alpha must be finite and in (0, 1): uniformity_alpha={uniformity_alpha}")]
    InvalidUniformityAlpha { uniformity_alpha: f64 },

    /// A produced p-value was non-finite or outside `[0, 1]`.
    #[error(
        "trial produced out-of-range p-value: case={case}, trial_index={trial_index}, value={value}"
    )]
    OutOfRangePValue {
        case: String,
        trial_index: usize,
        value: f64,
    },

    /// A test case name was empty or whitespace-only.
    #[error("test name must not be empty")]
    EmptyCaseName,

    /// Two cases in the same suite shared a name.
    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },
}

fn validate_case_name(name: String) -> Result<String, ParanoidError> {
    if name.trim().is_empty() {
        return Err(ParanoidError::EmptyCaseName);
    }
    Ok(name)
}

/// Converts a two-sided z-score (as produced by [`crate::testing::ChiSqResult`],
/// [`crate::testing::SerialResult`], or [`crate::testing::RunsResult`]) into an
/// approximate p-value, suitable as a trial input to [`run_paranoid`].
pub fn p_value_from_z(z_score: f64) -> f64 {
    erfc(z_score.abs() / std::f64::consts::SQRT_2)
}

/// Runs a paranoid (battery-level) meta-test: calls `trial` once per configured
/// trial count, where `trial(i)` must return the p-value of the `i`-th
/// independent run of the underlying test, then applies the NIST SP 800-22
/// §4.2 proportion-of-passes and p-value-uniformity checks.
pub fn run_paranoid(
    name: impl Into<String>,
    config: ParanoidConfig,
    trial: &mut dyn FnMut(usize) -> f64,
) -> Result<ParanoidResult, ParanoidError> {
    config.validate()?;
    let name = validate_case_name(name.into())?;

    let mut p_values = Vec::with_capacity(config.trials);
    for trial_index in 0..config.trials {
        let p = trial(trial_index);
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ParanoidError::OutOfRangePValue {
                case: name,
                trial_index,
                value: p,
            });
        }
        p_values.push(p);
    }

    let passes = p_values
        .iter()
        .filter(|&&p| p >= config.trial_alpha)
        .count();
    let pass_proportion = passes as f64 / config.trials as f64;

    let p_hat = 1.0 - config.trial_alpha;
    let bound = 3.0 * (p_hat * (1.0 - p_hat) / config.trials as f64).sqrt();
    let proportion_lower = (p_hat - bound).max(0.0);
    let proportion_upper = (p_hat + bound).min(1.0);
    let proportion_verdict = if (proportion_lower..=proportion_upper).contains(&pass_proportion) {
        ParanoidVerdict::Pass
    } else {
        ParanoidVerdict::Fail
    };

    let mut sorted = p_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_f = config.trials as f64;
    let mut ks_d_statistic = 0.0f64;
    for (i, &x) in sorted.iter().enumerate() {
        let i_f = (i + 1) as f64;
        let d_plus = i_f / n_f - x;
        let d_minus = x - (i_f - 1.0) / n_f;
        ks_d_statistic = ks_d_statistic.max(d_plus).max(d_minus);
    }
    let ks_critical_value = (-0.5 * (config.uniformity_alpha / 2.0).ln()).sqrt() / n_f.sqrt();
    let uniformity_verdict = if ks_d_statistic <= ks_critical_value {
        ParanoidVerdict::Pass
    } else {
        ParanoidVerdict::Fail
    };

    let verdict = if proportion_verdict == ParanoidVerdict::Pass
        && uniformity_verdict == ParanoidVerdict::Pass
    {
        ParanoidVerdict::Pass
    } else {
        ParanoidVerdict::Fail
    };

    Ok(ParanoidResult {
        name,
        trials: config.trials,
        passes,
        pass_proportion,
        proportion_lower,
        proportion_upper,
        proportion_verdict,
        ks_d_statistic,
        ks_critical_value,
        uniformity_verdict,
        verdict,
    })
}

struct ParanoidCase<'a> {
    name: String,
    trial: Box<dyn FnMut(usize) -> f64 + 'a>,
}

/// A suite that runs multiple named paranoid meta-test cases and collects
/// their [`ParanoidResult`]s. Bit-width-agnostic: cases are just p-value
/// producing closures, regardless of which underlying `Rng32`/`Rng64` test
/// they wrap.
#[derive(Default)]
pub struct ParanoidSuite<'a> {
    config: ParanoidConfig,
    cases: Vec<ParanoidCase<'a>>,
}

impl<'a> ParanoidSuite<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: ParanoidConfig) -> Result<Self, ParanoidError> {
        config.validate()?;
        Ok(Self {
            config,
            cases: Vec::new(),
        })
    }

    pub fn config(&self) -> ParanoidConfig {
        self.config
    }

    pub fn set_config(&mut self, config: ParanoidConfig) -> Result<(), ParanoidError> {
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

    /// Registers a named trial closure: `trial(i)` must return the p-value of
    /// the `i`-th independent run of the underlying test.
    pub fn add_trial<F>(
        &mut self,
        name: impl Into<String>,
        trial: F,
    ) -> Result<&mut Self, ParanoidError>
    where
        F: FnMut(usize) -> f64 + 'a,
    {
        let name = validate_case_name(name.into())?;
        self.cases.push(ParanoidCase {
            name,
            trial: Box::new(trial),
        });
        Ok(self)
    }

    pub fn run(&mut self) -> Result<Vec<ParanoidResult>, ParanoidError> {
        let mut seen = HashSet::with_capacity(self.cases.len());
        let mut out = Vec::with_capacity(self.cases.len());
        for case in &mut self.cases {
            if !seen.insert(case.name.clone()) {
                return Err(ParanoidError::DuplicateCaseName {
                    name: case.name.clone(),
                });
            }
            let result = run_paranoid(case.name.clone(), self.config, case.trial.as_mut())?;
            out.push(result);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfmt19937;
    use crate::testing::chisq::ChiSq32;

    #[test]
    fn paranoid_passes_for_good_generator() {
        let config = ParanoidConfig {
            trials: 30,
            ..ParanoidConfig::default()
        };
        let mut trial = |i: usize| {
            let mut rng = Sfmt19937::new(i as u64 + 1);
            let z = ChiSq32::from_urng(&mut rng)
                .run(format!("trial-{i}"))
                .unwrap()
                .z_score;
            p_value_from_z(z)
        };
        let res = run_paranoid("Sfmt19937.paranoid_chisq", config, &mut trial).unwrap();
        assert_eq!(res.verdict, ParanoidVerdict::Pass);
    }

    #[test]
    fn paranoid_rejects_biased_generator() {
        let config = ParanoidConfig {
            trials: 20,
            ..ParanoidConfig::default()
        };
        // A trial closure that always reports a maximally-suspicious p-value
        // must fail both the proportion and uniformity checks.
        let mut trial = |_i: usize| 0.0f64;
        let res = run_paranoid("always-fail", config, &mut trial).unwrap();
        assert_eq!(res.verdict, ParanoidVerdict::Fail);
    }
}
