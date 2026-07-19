use colored::Colorize;
use criterion::measurement::{Measurement, WallTime};
use thousands::Separable;

use urng::wide::{
    Jsf32x4, Jsf32x8, Jsf32x16, Pcg32x4, Pcg32x8, Pcg32x16, Sfc32x4, Sfc32x8, Sfc32x16,
    SplitMix32x4, SplitMix32x8, SplitMix32x16, Xoroshiro64Ssx4, Xoroshiro64Ssx8, Xoroshiro64Ssx16,
    Xorshift32x4, Xorshift32x8, Xorshift32x16, Xorshift128x4, Xorshift128x8, Xorshift128x16,
    Xorwowx4, Xorwowx8, Xorwowx16, Xoshiro128Ppx4, Xoshiro128Ppx8, Xoshiro128Ppx16, Xoshiro128Ssx4,
    Xoshiro128Ssx8, Xoshiro128Ssx16,
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

fn print_group(results: &[(&str, f64)], elem_bytes: usize, ceiling_gbps: f64) {
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
        let gbps = gs * elem_bytes as f64;
        let pct = gbps / ceiling_gbps * 100.0;
        println!(
            "{:<16}: {} {} {} {} {} {}  {}",
            name.bright_green(),
            format!("{:5.2}", gs).bright_cyan().bold(),
            "GS/s".bright_black(),
            format!("{:5.1}", gbps).bright_white(),
            "GB/s".bright_black(),
            format!("{:3.0}%", pct).bright_magenta(),
            "ceil".bright_black(),
            bar_colored,
        );
    }
}

/// Measure peak throughput of a wide RNG filling `buf` in `size`-element
/// chunks (one `nextu` call each), best of RUNS runs.
fn measure_wide<F: FnMut(*mut u32)>(buf: &mut [u32], size: usize, mut f: F) -> f64 {
    let n = buf.len();
    let meter = WallTime;
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = meter.start();
        let mut p = buf.as_mut_ptr();
        let mut written = 0;
        while written < n {
            f(p);
            p = p.wrapping_add(size);
            written += size;
        }
        let elapsed = meter.end(start);
        let t = n as f64 / (meter.to_f64(&elapsed) / 1e9) / G;
        if t > best {
            best = t;
        }
    }
    best
}

/// Fill `buf` with `size`-element chunks (used to warm pages before measuring).
fn warm_wide<F: FnMut(*mut u32)>(buf: &mut [u32], size: usize, mut f: F) {
    let n = buf.len();
    let mut p = buf.as_mut_ptr();
    let mut written = 0;
    while written < n {
        f(p);
        p = p.wrapping_add(size);
        written += size;
    }
}

/// Parallel non-temporal memset: measures this machine's pure DRAM
/// write-bandwidth ceiling, against which every RNG is scored. Mirrors
/// `examples/cabi/test.rs`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nt_memset_chunk(chunk: &mut [u32]) {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm256_set1_epi32(0x9E37_79B9u32 as i32);
        let mut p = chunk.as_mut_ptr();
        let mut rem = chunk.len();
        while (p as usize & 31) != 0 && rem > 0 {
            *p = 1;
            p = p.add(1);
            rem -= 1;
        }
        while rem >= 32 {
            _mm256_stream_si256(p as *mut _, v);
            _mm256_stream_si256(p.add(8) as *mut _, v);
            _mm256_stream_si256(p.add(16) as *mut _, v);
            _mm256_stream_si256(p.add(24) as *mut _, v);
            p = p.add(32);
            rem -= 32;
        }
        while rem >= 8 {
            _mm256_stream_si256(p as *mut _, v);
            p = p.add(8);
            rem -= 8;
        }
        while rem > 0 {
            *p = 1;
            p = p.add(1);
            rem -= 1;
        }
        _mm_sfence();
    }
}

/// Measure the NT-store write ceiling in GB/s on a warmed buffer.
/// Uses AVX2 streaming stores across all cores when available, else a scalar
/// fill as a fallback baseline.
fn measure_write_ceiling(buf: &mut [u32]) -> f64 {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let meter = WallTime;
    let mut best = 0.0f64;
    for _ in 0..RUNS {
        let start = meter.start();
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") {
            let chunk = (buf.len() / n_threads.max(1)).max(1);
            let base_addr = buf.as_mut_ptr() as usize;
            let len = buf.len();
            std::thread::scope(|s| {
                for t in 0..n_threads {
                    let start_idx = t * chunk;
                    let end_idx = if t == n_threads - 1 {
                        len
                    } else {
                        (start_idx + chunk).min(len)
                    };
                    if start_idx >= end_idx {
                        continue;
                    }
                    let addr = base_addr + start_idx * 4;
                    let slen = end_idx - start_idx;
                    s.spawn(move || unsafe {
                        let p = addr as *mut u32;
                        nt_memset_chunk(std::slice::from_raw_parts_mut(p, slen));
                    });
                }
            });
        } else {
            buf.iter_mut().for_each(|x| *x = 0x9E37_79B9);
        }
        #[cfg(not(target_arch = "x86_64"))]
        buf.iter_mut().for_each(|x| *x = 0x9E37_79B9);
        let elapsed = meter.end(start);
        let gbps = (buf.len() * 4) as f64 / (meter.to_f64(&elapsed) / 1e9) / G;
        if gbps > best {
            best = gbps;
        }
    }
    best
}

/// Run correctness checks on a wide RNG: determinism, and value ranges for
/// `nextu`, `nextf`, `randi`, `randf`.
macro_rules! test_wide {
    ($name:ident, $size:expr) => {{
        let mut rng1 = $name::new(0);
        let mut rng2 = $name::new(0);
        assert_eq!(
            rng1.nextu(),
            rng2.nextu(),
            "{}: determinism (nextu)",
            stringify!($name)
        );
        assert_eq!(
            rng1.nextf(),
            rng2.nextf(),
            "{}: determinism (nextf)",
            stringify!($name)
        );

        let u = rng1.nextu();
        for v in u.iter() {
            assert!(*v != 0, "{}: nextu produced 0", stringify!($name));
        }

        let f = rng1.nextf();
        for v in f.iter() {
            assert!(
                (0.0..1.0).contains(v),
                "{}: nextf out of range: {}",
                stringify!($name),
                v
            );
        }

        let ri = rng1.randi(-10, 10);
        for v in ri.iter() {
            assert!(
                (*v >= -10 && *v <= 10),
                "{}: randi out of range: {}",
                stringify!($name),
                v
            );
        }

        let rf = rng1.randf(-5.0, 5.0);
        for v in rf.iter() {
            assert!(
                (*v >= -5.0 && *v < 5.0),
                "{}: randf out of range: {}",
                stringify!($name),
                v
            );
        }

        println!(
            "{} {} (x{})",
            "[OK]".bright_green(),
            stringify!($name),
            $size
        );
    }};
}

/// Benchmark `nextu` throughput of one wide RNG and record the result.
macro_rules! bench_wide {
    ($buf:ident, $results:ident, $size:expr, $name:ident) => {{
        let mut rng = $name::new(0);
        let gs = measure_wide(&mut $buf, $size, |p| {
            let arr = rng.nextu();
            unsafe {
                std::ptr::copy_nonoverlapping(arr.as_ptr(), p, $size);
            }
        });
        $results.push((stringify!($name), gs));
    }};
}

fn main() {
    println!(
        "Wide RNG correctness + throughput (N={}, best of {} runs)",
        N.separate_with_commas().bright_cyan().bold(),
        RUNS.to_string().bright_yellow(),
    );
    println!("{}", "─".repeat(72).bright_black());

    // --- Correctness ---
    println!("{}", "Correctness".bright_yellow().bold());
    test_wide!(SplitMix32x4, 4);
    test_wide!(SplitMix32x8, 8);
    test_wide!(SplitMix32x16, 16);
    test_wide!(Sfc32x4, 4);
    test_wide!(Sfc32x8, 8);
    test_wide!(Sfc32x16, 16);
    test_wide!(Jsf32x4, 4);
    test_wide!(Jsf32x8, 8);
    test_wide!(Jsf32x16, 16);
    test_wide!(Pcg32x4, 4);
    test_wide!(Pcg32x8, 8);
    test_wide!(Pcg32x16, 16);
    test_wide!(Xoroshiro64Ssx4, 4);
    test_wide!(Xoroshiro64Ssx8, 8);
    test_wide!(Xoroshiro64Ssx16, 16);
    test_wide!(Xorshift32x4, 4);
    test_wide!(Xorshift32x8, 8);
    test_wide!(Xorshift32x16, 16);
    test_wide!(Xorshift128x4, 4);
    test_wide!(Xorshift128x8, 8);
    test_wide!(Xorshift128x16, 16);
    test_wide!(Xorwowx4, 4);
    test_wide!(Xorwowx8, 8);
    test_wide!(Xorwowx16, 16);
    test_wide!(Xoshiro128Ppx4, 4);
    test_wide!(Xoshiro128Ppx8, 8);
    test_wide!(Xoshiro128Ppx16, 16);
    test_wide!(Xoshiro128Ssx4, 4);
    test_wide!(Xoshiro128Ssx8, 8);
    test_wide!(Xoshiro128Ssx16, 16);

    // --- Throughput ---
    println!("{}", "─".repeat(72).bright_black());
    println!("{}", "Throughput (nextu)".bright_yellow().bold());

    let mut buf = vec![0u32; N];
    // Warm pages with the widest generator before measuring.
    {
        let mut rng = SplitMix32x16::new(0);
        warm_wide(&mut buf, 16, |p| {
            let arr = rng.nextu();
            unsafe {
                std::ptr::copy_nonoverlapping(arr.as_ptr(), p, 16);
            }
        });
    }

    let ceiling_gbps = measure_write_ceiling(&mut buf);
    println!(
        "NT-store write ceiling: {} GB/s (= {:.2} GS/s u32)",
        format!("{:.1}", ceiling_gbps).bright_magenta().bold(),
        ceiling_gbps / 4.0,
    );
    println!("{}", "─".repeat(72).bright_black());

    let mut results = Vec::new();
    bench_wide!(buf, results, 4, SplitMix32x4);
    bench_wide!(buf, results, 8, SplitMix32x8);
    bench_wide!(buf, results, 16, SplitMix32x16);
    bench_wide!(buf, results, 4, Sfc32x4);
    bench_wide!(buf, results, 8, Sfc32x8);
    bench_wide!(buf, results, 16, Sfc32x16);
    bench_wide!(buf, results, 4, Jsf32x4);
    bench_wide!(buf, results, 8, Jsf32x8);
    bench_wide!(buf, results, 16, Jsf32x16);
    bench_wide!(buf, results, 4, Pcg32x4);
    bench_wide!(buf, results, 8, Pcg32x8);
    bench_wide!(buf, results, 16, Pcg32x16);
    bench_wide!(buf, results, 4, Xoroshiro64Ssx4);
    bench_wide!(buf, results, 8, Xoroshiro64Ssx8);
    bench_wide!(buf, results, 16, Xoroshiro64Ssx16);
    bench_wide!(buf, results, 4, Xorshift32x4);
    bench_wide!(buf, results, 8, Xorshift32x8);
    bench_wide!(buf, results, 16, Xorshift32x16);
    bench_wide!(buf, results, 4, Xorshift128x4);
    bench_wide!(buf, results, 8, Xorshift128x8);
    bench_wide!(buf, results, 16, Xorshift128x16);
    bench_wide!(buf, results, 4, Xorwowx4);
    bench_wide!(buf, results, 8, Xorwowx8);
    bench_wide!(buf, results, 16, Xorwowx16);
    bench_wide!(buf, results, 4, Xoshiro128Ppx4);
    bench_wide!(buf, results, 8, Xoshiro128Ppx8);
    bench_wide!(buf, results, 16, Xoshiro128Ppx16);
    bench_wide!(buf, results, 4, Xoshiro128Ssx4);
    bench_wide!(buf, results, 8, Xoshiro128Ssx8);
    bench_wide!(buf, results, 16, Xoshiro128Ssx16);
    print_group(&results, 4, ceiling_gbps);
}
