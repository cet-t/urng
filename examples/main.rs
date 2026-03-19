use colored::Colorize;
use std::time::Instant;
use thousands::Separable;
#[allow(unused_imports)]
use urng::{
    rng32::{
        mt19937_free, mt19937_new, mt19937_next_u32s, pcg32_free, pcg32_new, pcg32_next_u32s,
        pcg32simd_free, pcg32simd_new, pcg32simd_next_u32s, pcg32x8_free, pcg32x8_new,
        pcg32x8_next_u32s, philox32_free, philox32_new, philox32_next_u32s, philox32x4_free,
        philox32x4_new, philox32x4_next_u32s, philox32x4x4_free, philox32x4x4_new,
        philox32x4x4_next_u32s, sfmt19937_free, sfmt19937_new, sfmt19937_next_u32s,
        splitmix32_free, splitmix32_new, splitmix32_next_u32s, splitmix32simd_free,
        splitmix32simd_new, splitmix32simd_next_u32s, splitmix32x16_free, splitmix32x16_new,
        splitmix32x16_next_u32s, squares32_free, squares32_new, squares32_next_u32s,
        squares32simd_free, squares32simd_new, squares32simd_next_u32s, squares32x8_free,
        squares32x8_new, squares32x8_next_u32s, threefry32x2_free, threefry32x2_new,
        threefry32x2_next_u32s, threefry32x4_free, threefry32x4_new, threefry32x4_next_u32s,
    },
    rng64::{
        mt1993764_free, mt1993764_new, mt1993764_next_u64s, philox64_free, philox64_new,
        philox64_next_u64s, sfmt1993764_free, sfmt1993764_new, sfmt1993764_next_u64s,
        splitmix64_free, splitmix64_new, splitmix64_next_u64s, threefish256_free, threefish256_new,
        threefish256_next_u64s, xoshiro256pp_free, xoshiro256pp_new, xoshiro256pp_next_u64s,
        xoshiro256ss_free, xoshiro256ss_new, xoshiro256ss_next_u64s, xoshiro256ssx2_free,
        xoshiro256ssx2_new, xoshiro256ssx2_next_u64s,
    },
};

// philox32x4x4    : 6.80 GS/s
// philox32x4      : 5.40 GS/s
// philox32        : 6.75 GS/s
// threefry32x4    : 5.28 GS/s
// threefry32x2    : 4.62 GS/s
// squares32       : 5.59 GS/s
// squares32x8     : 6.70 GS/s
// squares32simd   : 6.72 GS/s
// pcg32           : 1.06 GS/s
// pcg32x8         : 6.71 GS/s
// pcg32simd       : 6.76 GS/s
// splitmix32      : 2.33 GS/s
// splitmix32x16   : 6.72 GS/s
// splitmix32simd  : 6.68 GS/s

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
