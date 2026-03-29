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
        sfmt11213_free, sfmt11213_new, sfmt11213_next_u32s,
        sfmt1279_free, sfmt1279_new, sfmt1279_next_u32s,
        sfmt132049_free, sfmt132049_new, sfmt132049_next_u32s,
        sfmt19937_free, sfmt19937_new, sfmt19937_next_u32s,
        sfmt216091_free, sfmt216091_new, sfmt216091_next_u32s,
        sfmt2203_free, sfmt2203_new, sfmt2203_next_u32s,
        sfmt4253_free, sfmt4253_new, sfmt4253_next_u32s,
        sfmt44497_free, sfmt44497_new, sfmt44497_next_u32s,
        sfmt521_free, sfmt521_new, sfmt521_next_u32s,
        sfmt86243_free, sfmt86243_new, sfmt86243_next_u32s,
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
    },
    cabi64::{
        mt1993764_free, mt1993764_new, mt1993764_next_u64s, 
        philox64_free, philox64_new, philox64_next_u64s, 
        sfmt1993764_free, sfmt1993764_new, sfmt1993764_next_u64s,
        splitmix64_free, splitmix64_new, splitmix64_next_u64s, 
        threefish256_free, threefish256_new, threefish256_next_u64s, 
        xoshiro256pp_free, xoshiro256pp_new, xoshiro256pp_next_u64s,
        xoshiro256ss_free, xoshiro256ss_new, xoshiro256ss_next_u64s, 
        xoshiro256ssx2_free, xoshiro256ssx2_new, xoshiro256ssx2_next_u64s,
    },
};

// philox32x4x4    : 6.80 GS/s
// philox32x4      : 5.40 GS/s
// philox32        : 6.80 GS/s
// threefry32x4    : 5.28 GS/s
// threefry32x2    : 4.62 GS/s
// squares32       : 5.59 GS/s
// squares32x8     : 6.70 GS/s
// squares32simd   : 6.72 GS/s
// pcg32           : 1.06 GS/s
// pcg32x8         : 6.82 GS/s
// pcg32simd       : 6.84 GS/s
// splitmix32      : 2.33 GS/s
// splitmix32x16   : 6.80 GS/s
// splitmix32simd  : 6.68 GS/s
// mt19937         : 0.84 GS/s
// sfmt19937       : 0.88 GS/s
// xoshiro128pp    : 0.81 GS/s
// xoshiro128ppx16 : 6.82 GS/s
// xoshiro128ssx16 : 6.80 GS/s
// philox64        : 3.01 GS/s
// splitmix64      : 3.17 GS/s
// mt1993764       : 0.58 GS/s
// sfmt1993764     : 0.53 GS/s
// xoshiro256pp    : 3.13 GS/s
// xoshiro256ss    : 3.14 GS/s
// xoshiro256ssx2  : 3.40 GS/s

const N: usize = 100_000_000;
const G: f64 = 1_000_000_000f64;

#[allow(unused_macros)]
macro_rules! bench32 {
    ($new:expr, $next:expr, $free:expr $(,)?) => {{
        let name = stringify!($new).trim_end_matches("_new");
        let mut buffer = vec![0u32; N];
        let ptr = $new(0);

        let start = Instant::now();
        $next(ptr, buffer.as_mut_ptr(), N);
        let duration = start.elapsed();
        let thruput = N as f64 / duration.as_secs_f64() / G;
        println!(
            "{:<16}: {} {}",
            name.bright_green(),
            format!("{:.2}", thruput).bright_cyan().bold(),
            "GS/s".bright_black(),
        );
        $free(ptr);
    }};
}

#[allow(unused_macros)]
macro_rules! bench64 {
    ($new:expr, $next:expr, $free:expr $(,)?) => {{
        const N64: usize = N / 2;

        let name = stringify!($new).trim_end_matches("_new");
        let mut buffer = vec![0u64; N64];
        let ptr = $new(0);

        let start = Instant::now();
        $next(ptr, buffer.as_mut_ptr(), N64);
        let duration = start.elapsed();
        let thruput = N64 as f64 / duration.as_secs_f64() / G;
        println!(
            "{:<16}: {} {}",
            name.bright_green(),
            format!("{:.2}", thruput).bright_cyan().bold(),
            "GS/s".bright_black(),
        );
        $free(ptr);
    }};
}

fn main() {
    println!(
        "Benchmarking Random Number Generators (N = {})",
        N.separate_with_commas().bright_cyan().bold()
    );

    bench32!(philox32x4x4_new, philox32x4x4_next_u32s, philox32x4x4_free);
    bench32!(philox32x4_new, philox32x4_next_u32s, philox32x4_free);
    bench32!(philox32_new, philox32_next_u32s, philox32_free);
    bench32!(threefry32x4_new, threefry32x4_next_u32s, threefry32x4_free);
    bench32!(threefry32x2_new, threefry32x2_next_u32s, threefry32x2_free);
    bench32!(squares32_new, squares32_next_u32s, squares32_free);
    bench32!(squares32x8_new, squares32x8_next_u32s, squares32x8_free);
    bench32!(
        squares32simd_new,
        squares32simd_next_u32s,
        squares32simd_free
    );
    bench32!(pcg32_new, pcg32_next_u32s, pcg32_free);
    bench32!(pcg32x8_new, pcg32x8_next_u32s, pcg32x8_free);
    bench32!(pcg32simd_new, pcg32simd_next_u32s, pcg32simd_free);
    bench32!(splitmix32_new, splitmix32_next_u32s, splitmix32_free);
    bench32!(
        splitmix32x16_new,
        splitmix32x16_next_u32s,
        splitmix32x16_free
    );
    bench32!(
        splitmix32simd_new,
        splitmix32simd_next_u32s,
        splitmix32simd_free
    );
    bench32!(mt19937_new, mt19937_next_u32s, mt19937_free);
    bench32!(sfmt19937_new, sfmt19937_next_u32s, sfmt19937_free);
    bench32!(sfmt521_new, sfmt521_next_u32s, sfmt521_free);
    bench32!(sfmt1279_new, sfmt1279_next_u32s, sfmt1279_free);
    bench32!(sfmt2203_new, sfmt2203_next_u32s, sfmt2203_free);
    bench32!(sfmt4253_new, sfmt4253_next_u32s, sfmt4253_free);
    bench32!(sfmt11213_new, sfmt11213_next_u32s, sfmt11213_free);
    bench32!(sfmt44497_new, sfmt44497_next_u32s, sfmt44497_free);
    bench32!(sfmt86243_new, sfmt86243_next_u32s, sfmt86243_free);
    bench32!(sfmt132049_new, sfmt132049_next_u32s, sfmt132049_free);
    bench32!(sfmt216091_new, sfmt216091_next_u32s, sfmt216091_free);
    bench32!(xoshiro128pp_new, xoshiro128pp_next_u32s, xoshiro128pp_free);
    bench32!(
        xoshiro128ppx16_new,
        xoshiro128ppx16_next_u32s,
        xoshiro128ppx16_free
    );
    bench32!(
        xoshiro128ssx16_new,
        xoshiro128ssx16_next_u32s,
        xoshiro128ssx16_free
    );

    bench64!(philox64_new, philox64_next_u64s, philox64_free);
    bench64!(splitmix64_new, splitmix64_next_u64s, splitmix64_free);
    bench64!(mt1993764_new, mt1993764_next_u64s, mt1993764_free);
    bench64!(sfmt1993764_new, sfmt1993764_next_u64s, sfmt1993764_free);
    // bench64!(threefish256_new, threefish256_next_u64s, threefish256_free);
    bench64!(xoshiro256pp_new, xoshiro256pp_next_u64s, xoshiro256pp_free);
    bench64!(xoshiro256ss_new, xoshiro256ss_next_u64s, xoshiro256ss_free);
    bench64!(
        xoshiro256ssx2_new,
        xoshiro256ssx2_next_u64s,
        xoshiro256ssx2_free
    );
}
