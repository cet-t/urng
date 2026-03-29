use anyhow::Result;
use std::io::Write;

/// Result of a Monte Carlo pi estimation.
pub struct TestResult {
    pub name: String,
    pub n_pairs: usize,
    pub pi_est: f64,
    pub error: f64,
    pub error_pct: f64,
    pub passed: bool,
}

/// Estimate π using Monte Carlo method.
///
/// `rng_fn` must return values in [0, 1).
/// Each pair (x, y) is tested against the unit quarter-circle.
/// Passes when the relative error is less than 0.1%.
pub fn run(name: &str, rng_fn: &mut dyn FnMut() -> f64, n_pairs: usize) -> TestResult {
    let mut inside = 0u64;
    for _ in 0..n_pairs {
        let x = rng_fn();
        let y = rng_fn();
        if x * x + y * y < 1.0 {
            inside += 1;
        }
    }
    let pi_est = 4.0 * inside as f64 / n_pairs as f64;
    let error = (pi_est - std::f64::consts::PI).abs();
    let error_pct = error / std::f64::consts::PI * 100.0;
    TestResult {
        name: name.to_string(),
        n_pairs,
        pi_est,
        error,
        error_pct,
        passed: error_pct < 0.1,
    }
}

/// Print results to stdout and write to a log file.
pub fn log(results: &[TestResult], path: &str) -> Result<()> {
    use colored::Colorize;

    let header = format!(
        "\n{}\n{:-<70}\n{:<24} {:>12}  {:>10}  {:>8}  {}\n{:-<70}",
        "=== Monte Carlo π Estimation ===",
        "",
        "Algorithm",
        "π estimate",
        "error",
        "err %",
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
            "{:<24} {:>12.9}  {:>10.8}  {:>7.4}%  {}",
            r.name, r.pi_est, r.error, r.error_pct, verdict_color
        );
        writeln!(
            file,
            "{:<24} {:>12.9}  {:>10.8}  {:>7.4}%  {}",
            r.name, r.pi_est, r.error, r.error_pct, verdict_plain
        )?;
    }

    let footer = format!(
        "{:-<70}\nπ = {:.9},  N pairs = {:>10}  (err% < 0.1% → PASS)\n",
        "",
        std::f64::consts::PI,
        results.first().map(|r| r.n_pairs).unwrap_or(0),
    );
    println!("{}", footer);
    writeln!(file, "{}", footer)?;

    Ok(())
}
