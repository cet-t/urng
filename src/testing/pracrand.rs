use super::stream::{ByteStream32, ByteStream64};
use crate::rng::{Rng32, Rng64};
use std::io::{self, ErrorKind, Read, Write};
use std::process::{Command, Stdio};

// ── data size ─────────────────────────────────────────────────────────────────

/// Amount of data PracRand should analyse before reporting.
pub enum DataSize {
    MB(u64),
    GB(u64),
    TB(u64),
}

impl DataSize {
    fn as_arg(&self) -> String {
        match self {
            DataSize::MB(n) => format!("{n}MB"),
            DataSize::GB(n) => format!("{n}GB"),
            DataSize::TB(n) => format!("{n}TB"),
        }
    }
}

// ── result ────────────────────────────────────────────────────────────────────

/// Result of a PracRand run.
pub struct TestResult {
    /// `true` when no "FAIL" marker appears in PracRand's output.
    pub passed: bool,
    /// Full stdout captured from `RNG_test`.
    pub output: String,
}

impl TestResult {
    /// Print the captured output to stdout.
    pub fn print(&self) {
        print!("{}", self.output);
    }
}

// ── internal ──────────────────────────────────────────────────────────────────

fn pracrand_binary() -> String {
    std::env::var("PRACRAND_PATH").unwrap_or_else(|_| "RNG_test".to_string())
}

/// Pipes `stream` into `RNG_test <mode> -tlmax <size>` and returns the result.
///
/// PracRand exits as soon as it reaches `-tlmax`, closing its stdin; the writer
/// loop detects the broken pipe and stops. PracRand's stdout is small (a few KB
/// of text), so reading it after the child exits is safe without a reader thread.
fn run_inner(mut stream: impl Read, mode: &str, size: DataSize) -> io::Result<TestResult> {
    let bin = pracrand_binary();
    let mut child = Command::new(&bin)
        .args([mode, "-tlmax", &size.as_arg()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("failed to spawn '{bin}' (set PRACRAND_PATH if not in PATH): {e}"),
            )
        })?;

    // Write RNG bytes to child stdin until PracRand exits (broken pipe).
    {
        let mut stdin = child.stdin.take().expect("stdin not captured");
        let mut buf = [0u8; 65536];
        loop {
            let n = stream.read(&mut buf)?;
            if n == 0 {
                break;
            }
            match stdin.write_all(&buf[..n]) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::BrokenPipe => break,
                Err(e) => return Err(e),
            }
        }
        // stdin dropped here → child receives EOF
    }

    let out = child.wait_with_output()?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let passed = !text.contains("FAIL");

    Ok(TestResult { passed, output: text })
}

// ── public API ────────────────────────────────────────────────────────────────

/// Run PracRand on a [`Rng32`] generator.
///
/// # Errors
/// Returns an error if `RNG_test` cannot be found or spawned.
///
/// # Example
/// ```no_run
/// use urng::prelude::*;
/// use urng::testing::pracrand::{self, DataSize};
///
/// let mut rng = Xoshiro256Pp::new(42);
/// let result = pracrand::run32(rng, DataSize::GB(1)).unwrap();
/// result.print();
/// assert!(result.passed);
/// ```
pub fn run32<R: Rng32>(rng: R, size: DataSize) -> io::Result<TestResult> {
    run_inner(ByteStream32::new(rng), "stdin32", size)
}

/// Run PracRand on a [`Rng64`] generator.
///
/// # Errors
/// Returns an error if `RNG_test` cannot be found or spawned.
///
/// # Example
/// ```no_run
/// use urng::prelude::*;
/// use urng::testing::pracrand::{self, DataSize};
///
/// let rng = Xoshiro256Pp::new(42);
/// let result = pracrand::run64(rng, DataSize::GB(1)).unwrap();
/// result.print();
/// assert!(result.passed);
/// ```
pub fn run64<R: Rng64>(rng: R, size: DataSize) -> io::Result<TestResult> {
    run_inner(ByteStream64::new(rng), "stdin64", size)
}
