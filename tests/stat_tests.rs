#![allow(deprecated)]
mod stat;

use anyhow::Result;
use std::fs;
use std::sync::OnceLock;
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use urng::rng::Rng32V512;
use urng::rng::{Rng32, Rng64};
#[cfg(feature = "simd")]
use urng::rng32::Sfc32x4;
#[cfg(all(feature = "simd", target_feature = "avx2"))]
use urng::rng32::Sfc32x8;
use urng::rng32::{
    Jsf32, Mt19937, Pcg32, Philox32x4, Sfc32, Sfmt607, Sfmt1279, Sfmt2281, Sfmt4253, Sfmt11213,
    Sfmt19937, Sfmt44497, Sfmt86243, Sfmt132049, Sfmt216091, SplitMix32, Squares32, Threefry32x2,
    Threefry32x4, Xorshift32, Xorshift128, Xorwow,
};
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use urng::rng32::{Jsf32x16, Sfc32x16, Xoroshiro64Ssx16};
use urng::rng64::{
    Biski64, Cet64, Cet256, Mt1993764, Philox64, Sfc64, Sfmt1993764, SplitMix64, Threefish256,
    TwistedGFSR, Xorshift64, Xoshiro256Pp, Xoshiro256Ss,
};
use urng::testing::{
    ChiSqConfig, ChiSqResult, ChiSqSuite32, ChiSqVerdict, McPiConfig, McPiResult, McPiSuite32,
    McPiVerdict,
};

use crate::stat::{chisq, monte_carlo, scatter};

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

#[cfg(all(feature = "simd", target_feature = "avx2"))]
fn gen_arr8f32<F: FnMut() -> [f32; 8]>(mut f: F) -> impl FnMut() -> f64 {
    let mut buf = [0f32; 8];
    let mut i = 4usize;
    move || {
        if i >= 8 {
            buf = f();
            i = 0;
        }
        let v = buf[i] as f64;
        i += 1;
        v
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
fn gen_arr16f32<F: FnMut() -> [f32; 16]>(mut f: F) -> impl FnMut() -> f64 {
    let mut buf = [0f32; 16];
    let mut i = 16usize;
    move || {
        if i >= 16 {
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

// ── macros ────────────────────────────────────────────────────────────────────

// chi-square: always convert via nextu() for maximum precision.
macro_rules! push_chi {
    (s32$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (s64$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr2f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr2f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr4f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr4f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr8f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr8f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr16f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr16f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr2u64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr2u64(move || r.nextu());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr4f64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr4f64(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (nf64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextf();
        $rs.add_sampler(stringify!($algo), f)?;
    }};
}

macro_rules! push_monte {
    (s32$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (s64$rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr2f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr2f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr4f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr4f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr8f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr8f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr16f32 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr16f32(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr2u64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr2u64(move || r.nextu());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (arr4f64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = gen_arr4f64(move || r.nextf());
        $rs.add_sampler(stringify!($algo), f)?;
    }};
    (nf64 $rs:expr, $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let f = move || r.nextf();
        $rs.add_sampler(stringify!($algo), f)?;
    }};
}

macro_rules! do_scatter {
    (s32$algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0));
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (s64$algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0));
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr2f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2f32(move || r.nextf());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr4f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f32(move || r.nextf());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr8f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr8f32(move || r.nextf());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr16f32 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr16f32(move || r.nextf());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr2u64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr2u64(move || r.nextu());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (arr4f64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = gen_arr4f64(move || r.nextf());
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
    (nf64 $algo:ident, $rng:expr) => {{
        let mut r = $rng;
        let mut f = || r.nextf();
        scatter::plot(
            stringify!($algo),
            &mut f,
            SCATTER_N,
            &scatter_path(stringify!($algo)),
        )?;
    }};
}

macro_rules! rng_ctor {
    ($a:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            $a::new(0)
        }
    }};
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
    (arr8f32 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr8f32 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr8f32 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr8f32 $algo, rng_ctor!($algo));
    }};
    (arr16f32 $chi:expr, $mc:expr, $algo:ident) => {{
        push_chi!(arr16f32 $chi, $algo, rng_ctor!($algo));
        push_monte!(arr16f32 $mc, $algo, rng_ctor!($algo));
        do_scatter!(arr16f32 $algo, rng_ctor!($algo));
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
    chi: Vec<ChiSqResult>,
    monte: Vec<McPiResult>,
}

#[allow(deprecated)]
fn build_suite() -> Result<Suite> {
    ensure_logs_dir()?;
    println!("\n=== Lag-1 Scatter Plots ===");

    let mut chi = ChiSqSuite32::with_config(ChiSqConfig {
        samples: CHI_N,
        bins: CHI_BINS,
        z_limit: 3.0,
    })?;
    let mut monte = McPiSuite32::with_config(McPiConfig {
        pairs: MONTE_CARLO_N,
        max_error_pct: 0.1,
    })?;

    // rng32  (nextu → u32)
    // Note: Lcg32/Lcg64 excluded — plain u32/u64 multiply overflows for large modulus.
    reg!(s32 chi, monte, SplitMix32);
    reg!(s32 chi, monte, Mt19937);
    reg!(s32 chi, monte, Sfmt607);
    reg!(s32 chi, monte, Sfmt1279);
    reg!(s32 chi, monte, Sfmt2281);
    reg!(s32 chi, monte, Sfmt4253);
    reg!(s32 chi, monte, Sfmt11213);
    reg!(s32 chi, monte, Sfmt19937);
    reg!(s32 chi, monte, Sfmt44497);
    reg!(s32 chi, monte, Sfmt86243);
    reg!(s32 chi, monte, Sfmt132049);
    reg!(s32 chi, monte, Sfmt216091);
    reg!(s32 chi, monte, Pcg32);
    reg!(s32 chi, monte, Philox32x4);
    reg!(s32 chi, monte, Xorshift32);
    reg!(s32 chi, monte, Xorshift128);
    reg!(s32 chi, monte, Xorwow);
    reg!(s32 chi, monte, Threefry32x4);
    reg!(s32 chi, monte, Threefry32x2);
    reg!(s32 chi, monte, Squares32);
    reg!(s32 chi, monte, Jsf32);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    reg!(arr16f32 chi, monte, Jsf32x16);
    reg!(s32 chi, monte, Sfc32);
    #[cfg(feature = "simd")]
    reg!(arr4f32 chi, monte, Sfc32x4);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    reg!(arr8f32 chi, monte, Sfc32x8);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    reg!(arr16f32 chi, monte, Sfc32x16);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    reg!(arr16f32 chi, monte, Xoroshiro64Ssx16);

    // rng64  (nextu → u64 scalar, except Threefish256 → [f64;4])
    reg!(s64 chi, monte, SplitMix64);
    reg!(s64 chi, monte, Mt1993764);
    reg!(s64 chi, monte, Sfmt1993764);
    reg!(s64 chi, monte, Philox64);
    reg!(s64 chi, monte, Sfc64);
    reg!(s64 chi, monte, Xorshift64);
    reg!(s64 chi, monte, Cet64);
    reg!(s64 chi, monte, Cet256);
    reg!(s64 chi, monte, Xoshiro256Pp);
    reg!(s64 chi, monte, Xoshiro256Ss);
    reg!(nf64 chi, monte, TwistedGFSR);
    reg!(arr4f64 chi, monte, Threefish256);
    reg!(s64 chi, monte, Biski64);

    let chi = chi.run()?;
    let monte = monte.run()?;

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
        .filter(|r| r.verdict != ChiSqVerdict::Pass)
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "chi-square FAILED: {:?}", failed);
}

#[test]
fn monte_carlo_pi() {
    let failed: Vec<_> = suite()
        .monte
        .iter()
        .filter(|r| r.verdict != McPiVerdict::Pass)
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "monte_carlo_pi FAILED: {:?}", failed);
}

#[test]
fn scatter_plots() {
    let _ = suite();
}
