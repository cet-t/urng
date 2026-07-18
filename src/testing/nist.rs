//! NIST SP 800-22 (rev. 1a) statistical test suite, bit-level subset.
//!
//! Implements five of the fifteen NIST tests, chosen as the ones that are
//! self-contained (no external reference data) and give the broadest
//! coverage per unit of implementation complexity:
//! * **Frequency (Monobit) Test** (§2.1)
//! * **Frequency Test within a Block** (§2.2)
//! * **Runs Test** (§2.3, bit-level, distinct from [`crate::testing::runs`] which
//!   operates on the `[0, 1)` float stream)
//! * **Test for the Longest Run of Ones in a Block** (§2.4, fixed at `M = 128`)
//! * **Cumulative Sums (Cusum) Test**, forward mode (§2.13)
//!
//! Each sub-test yields a p-value; per NIST §4.3, a sequence passes a
//! sub-test when `p_value >= alpha`.

use crate::rng::{Rng32, Rng64};
use crate::testing::_internal::{erfc, igamc, normal_cdf};
use std::collections::HashSet;
use thiserror::Error;

/// Configuration for the NIST SP 800-22 bit-level test suite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NistConfig {
    /// Total number of bits drawn from the generator per test run.
    pub bits: usize,
    /// Significance level; a sub-test passes when `p_value >= alpha`.
    pub alpha: f64,
    /// Block size `M` used by the Block Frequency Test (§2.2). NIST recommends `M >= 20`.
    pub block_freq_m: usize,
}

impl Default for NistConfig {
    /// Returns the default configuration: 1,000,000 bits, alpha 0.01, block size 128.
    fn default() -> Self {
        Self {
            bits: 1_000_000,
            alpha: 0.01,
            block_freq_m: 128,
        }
    }
}

impl NistConfig {
    /// Validates the configuration, returning a [`NistError`] describing the first problem found.
    fn validate(&self) -> Result<(), NistError> {
        // 128 is the minimum needed for the fixed-M=128 Longest-Run-of-Ones test.
        if self.bits < 128 {
            return Err(NistError::InvalidBits { bits: self.bits });
        }
        if !self.alpha.is_finite() || !(0.0..1.0).contains(&self.alpha) {
            return Err(NistError::InvalidAlpha { alpha: self.alpha });
        }
        if self.block_freq_m < 2 || self.block_freq_m > self.bits {
            return Err(NistError::InvalidBlockFreqM {
                block_freq_m: self.block_freq_m,
            });
        }
        Ok(())
    }
}

/// Outcome of a single NIST sub-test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NistVerdict {
    /// `p_value >= alpha`.
    Pass,
    /// `p_value < alpha`.
    Fail,
}

/// The individual NIST SP 800-22 sub-tests implemented by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NistTest {
    /// Frequency (Monobit) Test, §2.1.
    Frequency,
    /// Frequency Test within a Block, §2.2.
    BlockFrequency,
    /// Runs Test, §2.3.
    Runs,
    /// Test for the Longest Run of Ones in a Block (`M = 128`), §2.4.
    LongestRunOfOnes,
    /// Cumulative Sums (Cusum) Test, forward mode, §2.13.
    CumulativeSums,
}

/// Result of a single NIST sub-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NistItemResult {
    /// Which sub-test produced this result.
    pub test: NistTest,
    /// Computed p-value.
    pub p_value: f64,
    /// Pass/fail verdict at the configured `alpha`.
    pub verdict: NistVerdict,
}

/// Full result of one run of the NIST SP 800-22 bit-level suite.
#[derive(Debug, Clone, PartialEq)]
pub struct NistResult {
    /// Name of the test case.
    pub name: String,
    /// Number of bits drawn.
    pub bits: usize,
    /// Significance level used for every sub-test.
    pub alpha: f64,
    /// One result per sub-test, in the order listed by [`NistTest`].
    pub items: Vec<NistItemResult>,
    /// Overall verdict: [`NistVerdict::Pass`] iff every sub-test passed.
    pub verdict: NistVerdict,
}

/// Errors that can occur while configuring or running the NIST suite.
#[derive(Debug, Error)]
pub enum NistError {
    /// Fewer than 128 bits were requested (needed by the Longest-Run-of-Ones test).
    #[error("bits must be at least 128: bits={bits}")]
    InvalidBits { bits: usize },

    /// An out-of-range or non-finite significance level was supplied.
    #[error("alpha must be finite and in (0, 1): alpha={alpha}")]
    InvalidAlpha { alpha: f64 },

    /// An invalid block-frequency block size was supplied.
    #[error("block_freq_m must be in [2, bits]: block_freq_m={block_freq_m}")]
    InvalidBlockFreqM { block_freq_m: usize },

    /// A test case name was empty or whitespace-only.
    #[error("test name must not be empty")]
    EmptyCaseName,

    /// Two cases in the same suite shared a name.
    #[error("duplicate test name in suite: {name}")]
    DuplicateCaseName { name: String },
}

struct NistCase<'a> {
    name: String,
    word_sampler: Box<dyn FnMut() -> u64 + 'a>,
    word_bits: u32,
}

fn validate_case_name(name: String) -> Result<String, NistError> {
    if name.trim().is_empty() {
        return Err(NistError::EmptyCaseName);
    }
    Ok(name)
}

fn verdict_of(p_value: f64, alpha: f64) -> NistVerdict {
    if p_value >= alpha {
        NistVerdict::Pass
    } else {
        NistVerdict::Fail
    }
}

fn draw_bits(word_sampler: &mut dyn FnMut() -> u64, word_bits: u32, n: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(n);
    'outer: loop {
        let w = word_sampler();
        for i in (0..word_bits).rev() {
            bits.push(((w >> i) & 1) == 1);
            if bits.len() == n {
                break 'outer;
            }
        }
    }
    bits
}

fn frequency_test(bits: &[bool]) -> f64 {
    let n = bits.len() as f64;
    let s: i64 = bits.iter().map(|&b| if b { 1 } else { -1 }).sum();
    let s_obs = (s.abs() as f64) / n.sqrt();
    erfc(s_obs / std::f64::consts::SQRT_2)
}

fn block_frequency_test(bits: &[bool], m: usize) -> f64 {
    let num_blocks = bits.len() / m;
    let mut chi2 = 0.0;
    for i in 0..num_blocks {
        let ones = bits[i * m..(i + 1) * m].iter().filter(|&&b| b).count();
        let pi = ones as f64 / m as f64;
        let delta = pi - 0.5;
        chi2 += delta * delta;
    }
    chi2 *= 4.0 * m as f64;
    igamc(num_blocks as f64 / 2.0, chi2 / 2.0)
}

fn runs_test(bits: &[bool]) -> f64 {
    let n = bits.len() as f64;
    let ones = bits.iter().filter(|&&b| b).count();
    let pi = ones as f64 / n;
    if (pi - 0.5).abs() >= 2.0 / n.sqrt() {
        return 0.0;
    }
    let mut v = 1usize;
    for w in bits.windows(2) {
        if w[0] != w[1] {
            v += 1;
        }
    }
    let num = (v as f64 - 2.0 * n * pi * (1.0 - pi)).abs();
    let den = 2.0 * (2.0 * n).sqrt() * pi * (1.0 - pi);
    erfc(num / den)
}

fn longest_run_of_ones_test(bits: &[bool]) -> f64 {
    const M: usize = 128;
    const PI_CAT: [f64; 6] = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124];
    let num_blocks = bits.len() / M;
    let mut v_counts = [0u64; 6];
    for i in 0..num_blocks {
        let block = &bits[i * M..(i + 1) * M];
        let mut max_run = 0usize;
        let mut cur = 0usize;
        for &b in block {
            if b {
                cur += 1;
                max_run = max_run.max(cur);
            } else {
                cur = 0;
            }
        }
        let cat = match max_run {
            0..=4 => 0,
            5 => 1,
            6 => 2,
            7 => 3,
            8 => 4,
            _ => 5,
        };
        v_counts[cat] += 1;
    }
    let mut chi2 = 0.0;
    for k in 0..6 {
        let expected = num_blocks as f64 * PI_CAT[k];
        let delta = v_counts[k] as f64 - expected;
        chi2 += delta * delta / expected;
    }
    igamc(2.5, chi2 / 2.0)
}

fn cumulative_sums_test(bits: &[bool]) -> f64 {
    let n = bits.len() as i64;
    let mut s = 0i64;
    let mut z = 0i64;
    for &b in bits {
        s += if b { 1 } else { -1 };
        z = z.max(s.abs());
    }
    if z == 0 {
        return 1.0;
    }
    let n_f = n as f64;
    let z_f = z as f64;
    let sqrt_n = n_f.sqrt();

    let start1 = (-n / z + 1) / 4;
    let end1 = (n / z - 1) / 4;
    let mut sum1 = 0.0;
    let mut k = start1;
    while k <= end1 {
        sum1 += normal_cdf(((4 * k + 1) as f64 * z_f) / sqrt_n);
        sum1 -= normal_cdf(((4 * k - 1) as f64 * z_f) / sqrt_n);
        k += 1;
    }

    let start2 = (-n / z - 3) / 4;
    let end2 = (n / z - 1) / 4;
    let mut sum2 = 0.0;
    let mut k = start2;
    while k <= end2 {
        sum2 += normal_cdf(((4 * k + 3) as f64 * z_f) / sqrt_n);
        sum2 -= normal_cdf(((4 * k + 1) as f64 * z_f) / sqrt_n);
        k += 1;
    }

    (1.0 - sum1 + sum2).clamp(0.0, 1.0)
}

fn run_nist(
    name: String,
    word_sampler: &mut dyn FnMut() -> u64,
    word_bits: u32,
    config: NistConfig,
) -> Result<NistResult, NistError> {
    config.validate()?;
    let bits = draw_bits(word_sampler, word_bits, config.bits);

    let mut items = Vec::with_capacity(5);
    let mut push = |test: NistTest, p_value: f64| {
        items.push(NistItemResult {
            test,
            p_value,
            verdict: verdict_of(p_value, config.alpha),
        });
    };

    push(NistTest::Frequency, frequency_test(&bits));
    push(
        NistTest::BlockFrequency,
        block_frequency_test(&bits, config.block_freq_m),
    );
    push(NistTest::Runs, runs_test(&bits));
    push(NistTest::LongestRunOfOnes, longest_run_of_ones_test(&bits));
    push(NistTest::CumulativeSums, cumulative_sums_test(&bits));

    let verdict = if items.iter().all(|i| i.verdict == NistVerdict::Pass) {
        NistVerdict::Pass
    } else {
        NistVerdict::Fail
    };

    Ok(NistResult {
        name,
        bits: config.bits,
        alpha: config.alpha,
        items,
        verdict,
    })
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
                    let name = validate_case_name(name.into())?;
                    let mut sampler = || self.rng.nextu() as u64;
                    run_nist(name, &mut sampler, $bits, self.config)
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
                    let name = validate_case_name(name.into())?;
                    self.cases.push(NistCase {
                        name,
                        word_sampler: Box::new(move || rng.nextu() as u64),
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
                        let result = run_nist(
                            case.name.clone(),
                            case.word_sampler.as_mut(),
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
