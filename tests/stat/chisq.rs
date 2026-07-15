use anyhow::Result;
use std::io::Write;
use urng::testing::{ChiSqResult, ChiSqVerdict};

/// Print results to stdout and write to a log file.
pub fn log(results: &[ChiSqResult], path: &str) -> Result<()> {
    use colored::Colorize;

    let header = format!(
        "\n{}\n{:-<60}\n{:<24} {:>10}  {:>8}  {:>8}  {}\n{:-<60}",
        "=== Chi-Square Uniformity Test ===", "", "Algorithm", "χ²", "z-score", "df", "Verdict", ""
    );
    println!("{}", header);

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    writeln!(file, "{}", header)?;

    for r in results {
        let passed = r.verdict == ChiSqVerdict::Pass;
        let verdict_color = if passed {
            "PASS".green().bold().to_string()
        } else {
            "FAIL".red().bold().to_string()
        };
        let verdict_plain = if passed { "PASS" } else { "FAIL" };
        println!(
            "{:<24} {:>10.2}  {:>+8.3}  {:>8.0}  {}",
            r.name, r.chi2, r.z_score, r.df, verdict_color
        );
        writeln!(
            file,
            "{:<24} {:>10.2}  {:>+8.3}  {:>8.0}  {}",
            r.name, r.chi2, r.z_score, r.df, verdict_plain
        )?;
    }

    let footer = format!(
        "{:-<60}\nN = {:>10},  bins = {}  (|z| < 3.0 → PASS)\n",
        "",
        results.first().map(|r| r.samples).unwrap_or(0),
        results.first().map(|r| r.bins).unwrap_or(0),
    );
    println!("{}", footer);
    writeln!(file, "{}", footer)?;

    Ok(())
}
