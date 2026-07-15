use anyhow::Result;
use std::io::Write;
use urng::testing::{McPiResult, McPiVerdict};

/// Print results to stdout and write to a log file.
pub fn log(results: &[McPiResult], path: &str) -> Result<()> {
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
        let passed = r.verdict == McPiVerdict::Pass;
        let verdict_color = if passed {
            "PASS".green().bold().to_string()
        } else {
            "FAIL".red().bold().to_string()
        };
        let verdict_plain = if passed { "PASS" } else { "FAIL" };
        println!(
            "{:<24} {:>12.9}  {:>10.8}  {:>7.4}%  {}",
            r.name, r.pi_estimate, r.absolute_error, r.error_pct, verdict_color
        );
        writeln!(
            file,
            "{:<24} {:>12.9}  {:>10.8}  {:>7.4}%  {}",
            r.name, r.pi_estimate, r.absolute_error, r.error_pct, verdict_plain
        )?;
    }

    let footer = format!(
        "{:-<70}\nπ = {:.9},  N pairs = {:>10}  (err% < 0.1% → PASS)\n",
        "",
        std::f64::consts::PI,
        results.first().map(|r| r.pairs).unwrap_or(0),
    );
    println!("{}", footer);
    writeln!(file, "{}", footer)?;

    Ok(())
}
