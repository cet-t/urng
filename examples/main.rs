use std::hint::black_box;
use std::time::Instant;
use urng::rng32::{Mt19937, Sfmt19937};

const N: usize = 100_000_000;

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
}
