mod stat;

use anyhow::Result;
use stat::{chisq, monte_carlo, scatter};
use std::fs;
use std::sync::OnceLock;
use urng::rng32::*;
use urng::rng64::*;

const LOG_DIR: &str = "tests/logs";
const CHI_N: usize = 10_000_000;
const CHI_BINS: usize = 256;
const MONTE_CARLO_N: usize = 10_000_000;
const SCATTER_N: usize = 2_000;

fn ensure_logs_dir() -> Result<()> {
    fs::create_dir_all(LOG_DIR)?;
    Ok(())
}

fn logs_path(file_name: &str) -> String {
    format!("{}/{}", LOG_DIR, file_name)
}

fn to_snake_case(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(chars.len() + 8);
    for (i, ch) in chars.iter().copied().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                let prev = chars[i - 1];
                let next_lower = chars
                    .get(i + 1)
                    .copied()
                    .is_some_and(|c| c.is_ascii_lowercase());
                if prev.is_ascii_lowercase()
                    || prev.is_ascii_digit()
                    || (prev.is_ascii_uppercase() && next_lower)
                {
                    out.push('_');
                }
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn scatter_path(algo_name: &str) -> String {
    logs_path(&format!("scatter_{}.png", to_snake_case(algo_name)))
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn gen_arr2f32<F: FnMut() -> [f32; 2]>(mut f: F) -> impl FnMut() -> f64 {
    let mut buf = [0f32; 2];
    let mut i = 2usize;
    move || {
        if i >= 2 {
            buf = f();
            i = 0;
        }
        let v = buf[i] as f64;
        i += 1;
        v
    }
}

fn gen_arr4f32<F: FnMut() -> [f32; 4]>(mut f: F) -> impl FnMut() -> f64 {
    let mut buf = [0f32; 4];
    let mut i = 4usize;
    move || {
        if i >= 4 {
            buf = f();
            i = 0;
        }
        let v = buf[i] as f64;
        i += 1;
        v
    }
}

fn gen_arr4f64<F: FnMut() -> [f64; 4]>(mut f: F) -> impl FnMut() -> f64 {
    let mut buf = [0f64; 4];
    let mut i = 4usize;
    move || {
        if i >= 4 {
            buf = f();
            i = 0;
        }
        let v = buf[i];
        i += 1;
        v
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn gen_arr2u64<F: FnMut() -> [u64; 2]>(mut f: F) -> impl FnMut() -> f64 {
    const SCALE: f64 = 1.0 / (u64::MAX as f64 + 1.0);
    let mut buf = [0u64; 2];
    let mut i = 2usize;
    move || {
        if i >= 2 { buf = f(); i = 0; }
        let v = buf[i] as f64 * SCALE; i += 1; v
    }
}

// ── macros ────────────────────────────────────────────────────────────────────

// chi-square: always convert via nextu() for maximum precision.
macro_rules! push_chi {
    (s32$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (s64$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (arr2f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2f32(move || r.nextf());
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (arr4f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f32(move || r.nextf());
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (arr2u64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2u64(move || r.nextu());
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (arr4f64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f64(move || r.nextf());
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
    (nf64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextf();
        $rs.push(chisq::run(stringify!($algo), &mut f, CHI_N, CHI_BINS));
    }};
}

macro_rules! push_monte {
    (s32$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (s64$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (arr2f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2f32(move || r.nextf());
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (arr4f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f32(move || r.nextf());
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (arr2u64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2u64(move || r.nextu());
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (arr4f64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f64(move || r.nextf());
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
    (nf64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextf();
        $rs.push(monte_carlo::run(stringify!($algo), &mut f, MONTE_CARLO_N));
    }};
}

macro_rules! do_scatter {
    (s32$algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (s64$algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (arr2f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2f32(move || r.nextf());
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (arr4f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f32(move || r.nextf());
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (arr2u64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2u64(move || r.nextu());
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (arr4f64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f64(move || r.nextf());
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
    (nf64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextf();
        scatter::plot(stringify!($algo), &mut f, SCATTER_N, &scatter_path(stringify!($algo)))?;
    }};
}

macro_rules! rng_ctor {
    (TwistedGFSR) => { TwistedGFSR::new(TwistedGFSR::new_seed()) };
    // seed=0 is degenerate for some generators; use seed=1.
    (Pcg32)      => { Pcg32::new(1) };
    (Xorshift32) => { Xorshift32::new(1) };
    (Xorwow)     => { Xorwow::new(1) };
    ($a:ident)   => { $a::new(0) };
}

macro_rules! reg {
    (s32$chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(s32$chi, $algo, rng_ctor!($algo));
        push_monte!(s32$mc, $algo, rng_ctor!($algo));
        do_scatter!(s32$algo, rng_ctor!($algo));
    }};
    (s64$chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(s64$chi, $algo, rng_ctor!($algo));
        push_monte!(s64$mc, $algo, rng_ctor!($algo));
        do_scatter!(s64$algo, rng_ctor!($algo));
    }};
    (arr2f32 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr2f32 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr2f32 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr2f32 $algo, rng_ctor!($algo));
    }};
    (arr4f32 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr4f32 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr4f32 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr4f32 $algo, rng_ctor!($algo));
    }};
    (arr2u64 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr2u64 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr2u64 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr2u64 $algo, rng_ctor!($algo));
    }};
    (arr4f64 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr4f64 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr4f64 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr4f64 $algo, rng_ctor!($algo));
    }};
    (nf64 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(nf64 $chi, $algo, rng_ctor!($algo));
        push_monte!(nf64 $mc, $algo, rng_ctor!($algo));
        do_scatter!(nf64 $algo, rng_ctor!($algo));
    }};
}

// ── suite ─────────────────────────────────────────────────────────────────────

struct Suite {
    chi: Vec<chisq::TestResult>,
    monte: Vec<monte_carlo::TestResult>,
}

#[allow(deprecated)]
fn build_suite() -> Result<Suite> {
    ensure_logs_dir()?;
    println!("\n=== Lag-1 Scatter Plots ===");

    let mut chi = Vec::new();
    let mut monte = Vec::new();

    // rng32  (nextu → u32)
    // Note: Lcg32/Lcg64 excluded — plain u32/u64 multiply overflows for large modulus.
    reg!(s32 chi, monte, SplitMix32);
    reg!(s32 chi, monte, Mt19937);
    reg!(s32 chi, monte, Sfmt19937);
    reg!(s32 chi, monte, Pcg32);
    reg!(arr4f32 chi, monte, Philox32x4);
    reg!(s32 chi, monte, Xorshift32);
    reg!(s32 chi, monte, Xorwow);
    reg!(arr4f32 chi, monte, Threefry32x4);
    reg!(arr2f32 chi, monte, Threefry32x2);
    reg!(s32 chi, monte, Squares32);

    // rng64  (nextu → u64 scalar, except Philox64 → [u64;2] and Threefish256 → [f64;4])
    reg!(s64 chi, monte, SplitMix64);
    reg!(s64 chi, monte, Mt1993764);
    reg!(s64 chi, monte, Sfmt1993764);
    reg!(arr2u64 chi, monte, Philox64);
    reg!(s64 chi, monte, Sfc64);
    reg!(s64 chi, monte, Xorshift64);
    reg!(s64 chi, monte, Cet64);
    reg!(s64 chi, monte, Xoshiro256Pp);
    reg!(s64 chi, monte, Xoshiro256Ss);
    reg!(nf64 chi, monte, TwistedGFSR);
    reg!(arr4f64 chi, monte, Threefish256);

    chisq::log(&chi, &logs_path("chi_square.log"))?;
    monte_carlo::log(&monte, &logs_path("monte_carlo.log"))?;

    Ok(Suite { chi, monte })
}

fn suite() -> &'static Suite {
    static SUITE: OnceLock<Suite> = OnceLock::new();
    SUITE.get_or_init(|| build_suite().expect("stat suite failed"))
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn chi_square() {
    let failed: Vec<_> = suite()
        .chi
        .iter()
        .filter(|r| !r.passed)
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "chi-square FAILED: {:?}", failed);
}

#[test]
fn monte_carlo_pi() {
    let failed: Vec<_> = suite()
        .monte
        .iter()
        .filter(|r| !r.passed)
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "monte_carlo_pi FAILED: {:?}", failed);
}

#[test]
fn scatter_plots() {
    let _ = suite();
}
