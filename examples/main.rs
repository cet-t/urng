use std::hint::black_box;
use std::time::Instant;
use urng::rng32::{
    Philox32x4, Philox32x4x4, philox32_free, philox32_new, philox32_next_u32s, philox32x4_free,
    philox32x4_new, philox32x4_next_u32s, philox32x4x4_free, philox32x4x4_new,
    philox32x4x4_next_u32s, squares32_free, squares32_new, squares32_next_u32s, threefry32x2_free,
    threefry32x2_new, threefry32x2_next_u32s, threefry32x4_free, threefry32x4_new,
    threefry32x4_next_u32s,
};

// 32x4(scalar) vs 32x4x4(vector)
// ~41.6 MB	    10.4M	0.96x (lose)
//  ~128 MB	      32M	1.13x (win)
//  ~400 MB	     100M	1.17x (win)
// 500M
// Philox32x4x4 Pure        : 2.99 GS
// Philox32x4 Pure          : 0.75 GS
// Philox32x4x4 CABI(rayon) : 6.68 GS
// Philox32x4 CABI(rayon)   : 5.77 GS
// Philox32 CABI(rayon)     : 6.80 GS
// Threefry32x4 CABI(rayon) : 5.35 GS
const N: usize = 500_000_000;

const G: f64 = 1_000_000_000f64;

fn main() {
    println!("Benchmarking Random Number Generators (N = {})", N);
    benchmark_throughput_philox32x4x4(N);
    benchmark_throughput_philox32x4(N);
    benchmark_throughput_philox32x4x4_cabi(N);
    benchmark_throughput_philox32x4_cabi(N);
    benchmark_throughput_philox32_cabi(N);
    benchmark_throughput_threefry32_cabi(N);
    benchmark_throughput_threefry32x2_cabi(N);
    benchmark_throughput_squares32_cabi(N);
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
