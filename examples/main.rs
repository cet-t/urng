use std::hint::black_box;
use std::time::Instant;
use thousands::Separable;
use urng::rng32::{
    Philox32x4, Philox32x4x4, pcg32_free, pcg32_new, pcg32_next_u32s, pcg32simd_free,
    pcg32simd_new, pcg32simd_next_u32s, pcg32x8_free, pcg32x8_new, pcg32x8_next_u32s,
    philox32_free, philox32_new, philox32_next_u32s, philox32x4_free, philox32x4_new,
    philox32x4_next_u32s, philox32x4x4_free, philox32x4x4_new, philox32x4x4_next_u32s,
    splitmix32_free, splitmix32_new, splitmix32_next_u32s, splitmix32simd_free, splitmix32simd_new,
    splitmix32simd_next_u32s, splitmix32x16_free, splitmix32x16_new, splitmix32x16_next_u32s,
    squares32_free, squares32_new, squares32_next_u32s, squares32simd_free, squares32simd_new,
    squares32simd_next_u32s, squares32x8_free, squares32x8_new, squares32x8_next_u32s,
    threefry32x2_free, threefry32x2_new, threefry32x2_next_u32s, threefry32x4_free,
    threefry32x4_new, threefry32x4_next_u32s,
};

// 32x4(scalar) vs 32x4x4(vector)
// ~41.6 MB	    10.4M	0.96x (lose)
//  ~128 MB	      32M	1.13x (win)
//  ~400 MB	     100M	1.17x (win)
// 500M
// Philox32x4x4 Pure        : 2.99 GS
// Philox32x4 Pure          : 0.75 GS
// Philox32x4x4 CABI        : 6.68 GS
// Philox32x4 CABI          : 5.77 GS
// Philox32 CABI            : 6.80 GS
// Threefry32x4 CABI        : 5.35 GS
// Squares32 CABI           : 5.45 GS
// Squares32x8 CABI         : 6.82 GS
// Squares32Simd CABI       : 6.88 GS
// Pcg32 CABI               : 0.90 GS
// Pcg32x8 CABI             : 6.76 GS
// Pcg32simd CABI           : 6.77 GS
// SplitMix32 CABI          : 1.11 GS
// SplitMix32x16 CABI       : 6.76 GS
// SplitMix32simd CABI      : 6.75 GS
const N: usize = 500_000_000;

const G: f64 = 1_000_000_000f64;

fn main() {
    println!(
        "Benchmarking Random Number Generators (N = {})",
        N.separate_with_commas()
    );

    // warm up memory pages to avoid first-touch page fault overhead
    {
        let mut warmup = vec![0u32; N];
        warmup.iter_mut().for_each(|v| *v = 1);
        black_box(&warmup);
    }

    benchmark_throughput_philox32x4x4(N);
    benchmark_throughput_philox32x4(N);
    benchmark_throughput_philox32x4x4_cabi(N);
    benchmark_throughput_philox32x4_cabi(N);
    benchmark_throughput_philox32_cabi(N);
    benchmark_throughput_threefry32_cabi(N);
    benchmark_throughput_threefry32x2_cabi(N);
    benchmark_throughput_squares32_cabi(N);
    benchmark_throughput_squares32x8_cabi(N);
    benchmark_throughput_squares32simd_cabi(N);
    benchmark_throughput_pcg32_cabi(N);
    benchmark_throughput_pcg32x8_cabi(N);
    benchmark_throughput_pcg32simd_cabi(N);
    benchmark_throughput_splitmix32_cabi(N);
    benchmark_throughput_splitmix32x16_cabi(N);
    benchmark_throughput_splitmix32simd_cabi(N);
}

fn benchmark_throughput_philox32x4x4(n: usize) {
    let mut rng = unsafe { Philox32x4x4::new(0) };
    let d = 16;

    let start = Instant::now();
    for _ in 0..n / d {
        unsafe {
            black_box(rng.nextu());
        }
    }
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Philox32x4x4 Pure Throughput: {:.2} GS", throughput);
}

fn benchmark_throughput_philox32x4(n: usize) {
    let mut rng = Philox32x4::new(0);
    let start = Instant::now();

    for _ in 0..n / 4 {
        black_box(rng.nextu());
    }
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Philox32x4 Pure Throughput: {:.2} GS", throughput);
}

fn benchmark_throughput_philox32x4x4_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = philox32x4x4_new(0);
    let start = Instant::now();
    philox32x4x4_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Philox32x4x4 CABI Throughput: {:.2} GS", throughput);
    philox32x4x4_free(ptr);
}

fn benchmark_throughput_philox32x4_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = philox32x4_new(0);
    let start = Instant::now();
    philox32x4_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Philox32x4 CABI Throughput: {:.2} GS", throughput);
    philox32x4_free(ptr);
}

fn benchmark_throughput_philox32_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = philox32_new(0);
    let start = Instant::now();
    philox32_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Philox32 CABI Throughput: {:.2} GS", throughput);
    philox32_free(ptr);
}

fn benchmark_throughput_threefry32_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = threefry32x4_new(0);
    let start = Instant::now();
    threefry32x4_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Threefry CABI Throughput: {:.2} GS", throughput);
    threefry32x4_free(ptr);
}

fn benchmark_throughput_threefry32x2_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = threefry32x2_new(0);
    let start = Instant::now();
    threefry32x2_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Threefry32x2 CABI Throughput: {:.2} GS", throughput);
    threefry32x2_free(ptr);
}

fn benchmark_throughput_squares32_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = squares32_new(0);
    let start = Instant::now();
    squares32_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Squares32 CABI Throughput: {:.2} GS", throughput);
    squares32_free(ptr);
}

fn benchmark_throughput_squares32x8_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = squares32x8_new(0);
    let start = Instant::now();
    squares32x8_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Squares32x8 CABI Throughput: {:.2} GS", throughput);
    squares32x8_free(ptr);
}

fn benchmark_throughput_squares32simd_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = squares32simd_new(0);
    let start = Instant::now();
    squares32simd_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Squares32Simd CABI Throughput: {:.2} GS", throughput);
    squares32simd_free(ptr);
}

fn benchmark_throughput_pcg32_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = pcg32_new(0);
    let start = Instant::now();
    pcg32_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Pcg32 CABI Throughput: {:.2} GS", throughput);
    pcg32_free(ptr);
}

fn benchmark_throughput_pcg32x8_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = pcg32x8_new(0);
    let start = Instant::now();
    pcg32x8_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Pcg32x8 CABI Throughput: {:.2} GS", throughput);
    pcg32x8_free(ptr);
}

fn benchmark_throughput_pcg32simd_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = pcg32simd_new(0);
    let start = Instant::now();
    pcg32simd_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("Pcg32simd CABI Throughput: {:.2} GS", throughput);
    pcg32simd_free(ptr);
}

fn benchmark_throughput_splitmix32_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = splitmix32_new(0);
    let start = Instant::now();
    splitmix32_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("SplitMix32 CABI Throughput: {:.2} GS", throughput);
    splitmix32_free(ptr);
}

fn benchmark_throughput_splitmix32x16_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = splitmix32x16_new(0);
    let start = Instant::now();
    splitmix32x16_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("SplitMix32x16 CABI Throughput: {:.2} GS", throughput);
    splitmix32x16_free(ptr);
}

fn benchmark_throughput_splitmix32simd_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = splitmix32simd_new(0);
    let start = Instant::now();
    splitmix32simd_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / G;
    println!("SplitMix32Simd CABI Throughput: {:.2} GS", throughput);
    splitmix32simd_free(ptr);
}
