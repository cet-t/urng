use colored::Colorize;
use criterion::measurement::{Measurement, WallTime};
use std::hint::black_box;
use thousands::Separable;
use urng::*;

const N: usize = 100_000_000;
const G: f64 = 1_000_000_000f64;

/// Number of timed runs per algorithm; best (max throughput) is reported.
const RUNS: usize = 3;

/// Bar chart width in full-block characters.
const BAR_WIDTH: usize = 40;

/// Unicode fractional block characters (1/8 increments).
const BLOCKS: &[char] = &[' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

/// Build a proportional bar string scaled to `max_gs`.
fn make_bar(gs: f64, max_gs: f64) -> String {
    let total_eighths = ((gs / max_gs) * (BAR_WIDTH * 8) as f64).round() as usize;
    let full = total_eighths / 8;
    let frac = total_eighths % 8;
    let mut s = String::with_capacity(BAR_WIDTH + 4);
    for _ in 0..full {
        s.push('█');
    }
    if frac > 0 && full < BAR_WIDTH {
        s.push(BLOCKS[frac]);
    }
    s
}

fn print_group(results: &[(&str, f64)]) {
    let max_gs = results.iter().map(|(_, gs)| *gs).fold(0.0f64, f64::max);
    let hi = max_gs * 0.75;
    let mid = max_gs * 0.50;
    for (name, gs) in results {
        let bar = make_bar(*gs, max_gs);
        let bar_colored = if *gs >= hi {
            bar.bright_green()
        } else if *gs >= mid {
            bar.bright_yellow()
        } else {
            bar.bright_red()
        };
        println!(
            "{:<16}: {} {}  {}",
            name.bright_green(),
            format!("{:5.2}", gs).bright_cyan().bold(),
            "GS/s".bright_black(),
            bar_colored,
        );
    }
}

/// Measures scalar `Rng32::nextu()` throughput over `RUNS` iterations.
/// Each value is passed through `black_box` so the loop can't be optimized away.
fn measure32<R: Rng32>(rng: &mut R) -> f64 {
    let meter = WallTime;
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = meter.start();
        for _ in 0..N {
            black_box(rng.nextu());
        }
        let elapsed = meter.end(start);
        let t = N as f64 / (meter.to_f64(&elapsed) / 1e9) / G;
        if t > best {
            best = t;
        }
    }
    best
}

/// Measures scalar `Rng64::nextu()` throughput over `RUNS` iterations.
fn measure64<R: Rng64>(rng: &mut R) -> f64 {
    let meter = WallTime;
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = meter.start();
        for _ in 0..N {
            black_box(rng.nextu());
        }
        let elapsed = meter.end(start);
        let t = N as f64 / (meter.to_f64(&elapsed) / 1e9) / G;
        if t > best {
            best = t;
        }
    }
    best
}

macro_rules! bench32 {
    ($results:ident, $name:ident) => {{
        let mut rng = $name::new(0);
        let gs = measure32(&mut rng);
        $results.push((stringify!($name), gs));
    }};
    ($results:ident, $($name:ident),+ $(,)?) => {
        $(bench32!($results, $name);)+
    };
}

macro_rules! bench64 {
    ($results:ident, $name:ident) => {{
        let mut rng = $name::new(0);
        let gs = measure64(&mut rng);
        $results.push((stringify!($name), gs));
    }};
    ($results:ident, $($name:ident),+ $(,)?) => {
        $(bench64!($results, $name);)+
    };
}

fn main() {
    println!(
        "Benchmarking scalar Rng32/Rng64 implementations (N={}, best of {} runs)",
        N.separate_with_commas().bright_cyan().bold(),
        RUNS.to_string().bright_yellow(),
    );
    println!("{}", "─".repeat(72).bright_black());

    // --- 32-bit (Rng32::nextu() -> u32) ---
    // Lcg32 excluded: deprecated, fixed-parameter generator not meant for benchmarking.
    let mut r32 = Vec::new();
    bench32!(r32, Philox32x4, Threefry32x4, Threefry32x2);
    bench32!(r32, Squares32);
    bench32!(r32, Pcg32);
    bench32!(r32, SplitMix32);
    bench32!(r32, Mt19937, Sfmt19937);
    bench32!(
        r32, Sfmt607, Sfmt1279, Sfmt2281, Sfmt4253, Sfmt11213, Sfmt44497, Sfmt86243, Sfmt132049,
        Sfmt216091
    );
    bench32!(r32, Xoshiro128Pp, Xoshiro128Ss);
    bench32!(r32, Xoroshiro64Ss);
    bench32!(r32, Xorshift32, Xorshift128, Xorwow);
    bench32!(r32, Jsf32);
    bench32!(r32, Sfc32);
    print_group(&r32);

    println!("{}", "─".repeat(72).bright_black());

    // --- 64-bit (Rng64::nextu() -> u64) ---
    // Lcg64 excluded: deprecated, fixed-parameter generator not meant for benchmarking.
    let mut r64 = Vec::new();
    bench64!(r64, Philox64);
    bench64!(r64, SplitMix64);
    bench64!(r64, Cet64, Cet256);
    bench64!(r64, Mt1993764, Sfmt1993764);
    bench64!(r64, Xoshiro256Pp, Xoshiro256Ss);
    bench64!(r64, Xoroshiro128Pp, Xoroshiro128Ss);
    bench64!(r64, Xorshift64);
    bench64!(r64, TwistedGFSR);
    bench64!(r64, Sfc64);
    bench64!(r64, Biski64);
    print_group(&r64);
}
