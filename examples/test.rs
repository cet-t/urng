use colored::Colorize;
use std::time::Instant;
use thousands::Separable;
#[rustfmt::skip]
#[allow(unused_imports)]
use urng::{
    cabi32::{
        mt19937_free, mt19937_new, mt19937_next_u32s,
        pcg32_free, pcg32_new, pcg32_next_u32s,
        pcg32simd_free, pcg32simd_new, pcg32simd_next_u32s,
        pcg32x8_free, pcg32x8_new, pcg32x8_next_u32s,
        philox32_free, philox32_new, philox32_next_u32s,
        philox32x4_free, philox32x4_new, philox32x4_next_u32s,
        philox32x4x4_free, philox32x4x4_new, philox32x4x4_next_u32s,
        sfmt607_free, sfmt607_new, sfmt607_next_u32s,
        sfmt1279_free, sfmt1279_new, sfmt1279_next_u32s,
        sfmt2281_free, sfmt2281_new, sfmt2281_next_u32s,
        sfmt4253_free, sfmt4253_new, sfmt4253_next_u32s,
        sfmt11213_free, sfmt11213_new, sfmt11213_next_u32s,
        sfmt19937_free, sfmt19937_new, sfmt19937_next_u32s,
        sfmt44497_free, sfmt44497_new, sfmt44497_next_u32s,
        sfmt86243_free, sfmt86243_new, sfmt86243_next_u32s,
        sfmt132049_free, sfmt132049_new, sfmt132049_next_u32s,
        sfmt216091_free, sfmt216091_new, sfmt216091_next_u32s,
        splitmix32_free, splitmix32_new, splitmix32_next_u32s,
        splitmix32simd_free, splitmix32simd_new, splitmix32simd_next_u32s,
        splitmix32x16_free, splitmix32x16_new, splitmix32x16_next_u32s,
        squares32_free, squares32_new, squares32_next_u32s,
        squares32simd_free, squares32simd_new, squares32simd_next_u32s,
        squares32x8_free, squares32x8_new, squares32x8_next_u32s,
        threefry32x2_free, threefry32x2_new, threefry32x2_next_u32s,
        threefry32x4_free, threefry32x4_new, threefry32x4_next_u32s,
        xoshiro128pp_free, xoshiro128pp_new, xoshiro128pp_next_u32s,
        xoshiro128ppx16_free, xoshiro128ppx16_new, xoshiro128ppx16_next_u32s,
        xoshiro128ss_free, xoshiro128ss_new, xoshiro128ss_next_u32s,
        xoshiro128ssx16_free, xoshiro128ssx16_new, xoshiro128ssx16_next_u32s,
        jsf32_free, jsf32_new, jsf32_next_u32s,
        jsf32x16_free, jsf32x16_new, jsf32x16_next_u32s,
    },
    cabi64::{
        mt1993764_free, mt1993764_new, mt1993764_next_u64s,
        philox64_free, philox64_new, philox64_next_u64s,
        sfmt1993764_free, sfmt1993764_new, sfmt1993764_next_u64s,
        splitmix64_free, splitmix64_new, splitmix64_next_u64s,
        cet64_free, cet64_new, cet64_next_u64s,
        cet64x8_free, cet64x8_new, cet64x8_next_u64s,
        cet256_free, cet256_new, cet256_next_u64s,
        cet256x2_free, cet256x2_new, cet256x2_next_u64s,
        threefish256_free, threefish256_new, threefish256_next_u64s,
        xoshiro256pp_free, xoshiro256pp_new, xoshiro256pp_next_u64s,
        xoshiro256ss_free, xoshiro256ss_new, xoshiro256ss_next_u64s,
        xoshiro256ssx2_free, xoshiro256ssx2_new, xoshiro256ssx2_next_u64s,
        sfc64_free, sfc64_new, sfc64_next_u64s,
        sfc64x8_free, sfc64x8_new, sfc64x8_next_u64s,
        biski64_free, biski64_new, biski64_next_u64s,
        biski64x8_free, biski64x8_new, biski64x8_next_u64s,
    },
};

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

/// Measure peak throughput of a single RNG over RUNS iterations.
/// `buf` must already be page-faulted (warmed) before calling.
fn measure<F>(buf: &mut [u32], mut f: F) -> f64
where
    F: FnMut(*mut u32, usize),
{
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = Instant::now();
        f(buf.as_mut_ptr(), N);
        let t = N as f64 / start.elapsed().as_secs_f64() / G;
        if t > best {
            best = t;
        }
    }
    best
}

fn measure64<F>(buf: &mut [u64], mut f: F) -> f64
where
    F: FnMut(*mut u64, usize),
{
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = Instant::now();
        f(buf.as_mut_ptr(), N);
        let t = N as f64 / start.elapsed().as_secs_f64() / G;
        if t > best {
            best = t;
        }
    }
    best
}

fn print_group(results: &[(&str, f64)], hi: f64, mid: f64) {
    let max_gs = results.iter().map(|(_, gs)| *gs).fold(0.0f64, f64::max);
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
            format!("{:.2}", gs).bright_cyan().bold(),
            "GS/s".bright_black(),
            bar_colored,
        );
    }
}

macro_rules! bench32 {
    ($buf:ident, $results:ident, $name:ident) => {
        paste::paste! {
            let ptr = [<$name _new>](0);
            let gs = measure(&mut $buf, |p, n| [<$name _next_u32s>](ptr, p, n));
            [<$name _free>](ptr);
            $results.push((stringify!($name), gs));
        }
    };
    ($buf:ident, $results:ident, $($name:ident),+) => {
        $(bench32!($buf, $results, $name);)+
    };
}

macro_rules! bench64 {
    ($buf:ident, $results:ident, $name:ident) => {
        paste::paste! {
            let ptr = [<$name _new>](0);
            let gs = measure64(&mut $buf, |p, n| [<$name _next_u64s>](ptr, p, n));
            [<$name _free>](ptr);
            $results.push((stringify!($name), gs));
        }
    };
    ($buf:ident, $results:ident, $($name:ident),+) => {
        $(bench64!($buf, $results, $name);)+
    };
}

fn main() {
    println!(
        "Benchmarking RNGs (N={}, best of {} runs, warmed buffer)",
        N.separate_with_commas().bright_cyan().bold(),
        RUNS.to_string().bright_yellow(),
    );
    println!("{}", "─".repeat(72).bright_black());

    // --- 32-bit ---
    // Allocate once and touch every page to eliminate page-fault overhead.
    let mut buf32 = vec![0u32; N];
    // Warm: write to every cache line so pages are mapped before measurement.
    {
        let ptr = jsf32x16_new(0);
        jsf32x16_next_u32s(ptr, buf32.as_mut_ptr(), N);
        jsf32x16_free(ptr);
    }

    let mut r32: Vec<(&str, f64)> = Vec::new();
    bench32!(buf32, r32, philox32x4x4, philox32x4, philox32);
    bench32!(buf32, r32, threefry32x4, threefry32x2);
    bench32!(buf32, r32, squares32, squares32x8, squares32simd);
    bench32!(buf32, r32, pcg32, pcg32x8, pcg32simd);
    bench32!(buf32, r32, splitmix32, splitmix32x16, splitmix32simd);
    bench32!(buf32, r32, mt19937, sfmt19937);
    bench32!(
        buf32, r32,
        sfmt607, sfmt1279, sfmt2281, sfmt4253, sfmt11213,
        sfmt44497, sfmt86243, sfmt132049, sfmt216091
    );
    bench32!(buf32, r32, xoshiro128pp, xoshiro128ppx16, xoshiro128ssx16);
    bench32!(buf32, r32, jsf32, jsf32x16);
    print_group(&r32, 6.0, 4.0);

    println!("{}", "─".repeat(72).bright_black());

    // --- 64-bit ---
    let mut buf64 = vec![0u64; N];
    {
        use urng::cabi64::splitmix64_next_u64s;
        let ptr = splitmix64_new(0);
        splitmix64_next_u64s(ptr, buf64.as_mut_ptr(), N);
        splitmix64_free(ptr);
    }

    let mut r64: Vec<(&str, f64)> = Vec::new();
    bench64!(buf64, r64, philox64);
    bench64!(buf64, r64, splitmix64);
    bench64!(buf64, r64, cet64, cet64x8, cet256, cet256x2);
    bench64!(buf64, r64, mt1993764, sfmt1993764);
    bench64!(buf64, r64, threefish256);
    bench64!(buf64, r64, xoshiro256pp, xoshiro256ss, xoshiro256ssx2);
    bench64!(buf64, r64, sfc64, sfc64x8);
    bench64!(buf64, r64, biski64, biski64x8);
    print_group(&r64, 3.0, 2.0);
}
