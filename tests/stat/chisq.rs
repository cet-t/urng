use anyhow::Result;
use std::io::Write;

/// Result of a chi-square uniformity test.
pub struct TestResult {
    pub name:   String,
    pub n:      usize,
    pub bins:   usize,
    pub chi2:   f64,
    pub df:     f64,
    pub z:      f64,
    pub passed: bool,
}

/// Run a chi-square uniformity test.
///
/// `rng_fn` must return values in [0, 1).
/// Uses `bins` equally-spaced buckets; passes when |z| < 3.0
/// (chi2 within 3 standard deviations of the expected df).
pub fn run(name: &str, rng_fn: &mut dyn FnMut() -> f64, n: usize, bins: usize) -> TestResult {
    let mut counts = vec![0u64; bins];
    for _ in 0..n {
        let x = rng_fn();
        let bin = ((x * bins as f64) as usize).min(bins - 1);
        counts[bin] += 1;
    }
    let expected = n as f64 / bins as f64;
    let chi2: f64 = counts
        .iter()
        .map(|&c| {
            let d = c as f64 - expected;
            d * d / expected
        })
        .sum();
    let df = (bins - 1) as f64;
    let z = (chi2 - df) / (2.0 * df).sqrt();
    TestResult {
        name: name.to_string(),
        n,
        bins,
        chi2,
        df,
        z,
        passed: z.abs() < 3.0,
    }
}

/// Print results to stdout and write to a log file.
pub fn log(results: &[TestResult], path: &str) -> Result<()> {
    use colored::Colorize;

    let header = format!(
        "\n{}\n{:-<60}\n{:<24} {:>10}  {:>8}  {:>8}  {}\n{:-<60}",
        "=== Chi-Square Uniformity Test ===",
        "",
        "Algorithm",
        "χ²",
        "z-score",
        "df",
        "Verdict",
        ""
    );
    println!("{}", header);

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    writeln!(file, "{}", header)?;

    for r in results {
        let verdict_color = if r.passed {
            "PASS".green().bold().to_string()
        } else {
            "FAIL".red().bold().to_string()
        };
        let verdict_plain = if r.passed { "PASS" } else { "FAIL" };
        println!(
            "{:<24} {:>10.2}  {:>+8.3}  {:>8.0}  {}",
            r.name, r.chi2, r.z, r.df, verdict_color
        );
        writeln!(
            file,
            "{:<24} {:>10.2}  {:>+8.3}  {:>8.0}  {}",
            r.name, r.chi2, r.z, r.df, verdict_plain
        )?;
    }

    let footer = format!(
        "{:-<60}\nN = {:>10},  bins = {}  (|z| < 3.0 → PASS)\n",
        "",
        results.first().map(|r| r.n).unwrap_or(0),
        results.first().map(|r| r.bins).unwrap_or(0),
    );
    println!("{}", footer);
    writeln!(file, "{}", footer)?;

    Ok(())
}
