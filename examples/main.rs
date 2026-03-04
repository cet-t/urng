use std::hint::black_box;
use std::time::Instant;
use urng::rng32::{Mt19937, Philox32, Philox32_512, Sfmt19937};

const N: usize = 4 * 100_000_000;

fn main() {
    println!("Benchmarking Random Number Generators (N = {})", N);

    // Benchmark Mt19937
    let mut mt = Mt19937::new(12345);
    let start_mt = Instant::now();
    for _ in 0..N {
        black_box(mt.nextu());
    }
    let duration_mt = start_mt.elapsed();
    println!("Mt19937: {:?}", duration_mt);

    // Benchmark Sfmt19937
    let mut sfmt = Sfmt19937::new(12345);
    let start_sfmt = Instant::now();
    for _ in 0..N {
        black_box(sfmt.nextu());
    }
    let duration_sfmt = start_sfmt.elapsed();
    println!("Sfmt19937:  {:?}", duration_sfmt);

    // Calculate speedup
    let speedup = duration_mt.as_secs_f64() / duration_sfmt.as_secs_f64();
    println!("Sfmt speedup: {:.2}x", speedup);

    // Benchmark Philox32
    let mut philox32x4 = Philox32::new(12345);
    let start_philox32x4 = Instant::now();
    for _ in 0..N / 4 {
        black_box(philox32x4.nextu());
    }
    let duration_philox32x4 = start_philox32x4.elapsed();
    println!("Philox32x4: {:?}", duration_philox32x4);

    // Benchmark Philox32_512
    let mut philox32x4x4 = Philox32_512::new(12345);
    let start_philox32x4x4 = Instant::now();
    for _ in 0..N / 16 {
        black_box(philox32x4x4.nextu());
    }
    let duration_philox32x4x4 = start_philox32x4x4.elapsed();
    println!("Philox32x4x4: {:?}", duration_philox32x4x4);

    // Calculate speedup
    let speedup = duration_philox32x4.as_secs_f64() / duration_philox32x4x4.as_secs_f64();
    println!("Philox32x4-10x4 speedup: {:.2}x", speedup);
}
