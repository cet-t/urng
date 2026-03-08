use colored::Colorize;
use std::hint::black_box;
use std::time::Instant;
use urng::rng32::{
    Mt19937, Philox32x4, Philox32x4x4, Sfmt19937, philox32_free, philox32_new, philox32_rand_f32s,
    philox32x4_free, philox32x4_new, philox32x4_next_f32s, philox32x4_next_u32s,
    philox32x4_rand_f32s, philox32x4_rand_i32s, philox32x4x4_free, philox32x4x4_new,
    philox32x4x4_next_f32s, philox32x4x4_next_u32s, philox32x4x4_rand_f32s, philox32x4x4_rand_i32s,
};

// 32x4(scalar) vs 32x4x4(vector)
// ~41.6 MB	    10.4M	0.96x (lose)
//  ~128 MB	      32M	1.13x (win)
//  ~400 MB	     100M	1.17x (win)
const N: usize = 100_000_000;
const M: u32 = 10;

fn main() {
    println!("Benchmarking Random Number Generators (N = {})", N);

    /*     // Benchmark Mt19937
        let mut mt = Mt19937::new(12345);
        let start_mt = Instant::now();
        for _ in 0..N {
            black_box(mt.nextu());
        }
        let duration_mt = start_mt.elapsed();
        println!("{}: {:?}", "Mt19937".bright_green(), duration_mt);

        // Benchmark Sfmt19937
        let mut sfmt = Sfmt19937::new(12345);
        let start_sfmt = Instant::now();
        for _ in 0..N {
            black_box(sfmt.nextu());
        }
        let duration_sfmt = start_sfmt.elapsed();
        println!("{}: {:?}", "Sfmt19937".bright_green(), duration_sfmt);

        // Calculate speedup
        let speedup = duration_mt.as_secs_f64() / duration_sfmt.as_secs_f64();
        println!("{}: {:.2}x\n", "Sfmt speedup".bright_cyan(), speedup);

        // Benchmark Philox32
        let mut philox32x4 = Philox32x4::new(12345);
        let start_philox32x4 = Instant::now();
        for _ in 0..N / 4 {
            black_box(philox32x4.nextu());
        }
        let duration_philox32x4 = start_philox32x4.elapsed();
        println!("{}: {:?}", "Philox32x4".bright_green(), duration_philox32x4);

        // Benchmark Philox32_512
        let mut philox32x4x4 = Philox32x4x4::new(12345);
        let start_philox32x4x4 = Instant::now();
        for _ in 0..N / 16 {
            black_box(philox32x4x4.nextu());
        }
        let duration_philox32x4x4 = start_philox32x4x4.elapsed();
        println!(
            "{}: {:?}",
            "Philox32x4x4".bright_green(),
            duration_philox32x4x4
        );

        // Calculate speedup
        let speedup = duration_philox32x4.as_secs_f64() / duration_philox32x4x4.as_secs_f64();
        println!(
            "{}: {:.2}x\n",
            "Philox32x4-10x4 speedup".bright_cyan(),
            speedup
        );

        let start_philox32x4 = Instant::now();
        for _ in 0..N / 4 {
            black_box(philox32x4.nextf());
        }
        let duration_philox32x4 = start_philox32x4.elapsed();
        println!("{}: {:?}", "Philox32x4".bright_green(), duration_philox32x4);

        // Benchmark Philox32_512
        let start_philox32x4x4 = Instant::now();
        for _ in 0..N / 16 {
            black_box(philox32x4x4.nextf());
        }
        let duration_philox32x4x4 = start_philox32x4x4.elapsed();
        println!(
            "{}: {:?}",
            "Philox32x4x4".bright_green(),
            duration_philox32x4x4
        );

        // Calculate speedup
        let speedup = duration_philox32x4.as_secs_f64() / duration_philox32x4x4.as_secs_f64();
        println!(
            "{}: {:.2}x",
            "Philox32x4-10x4 speedup".bright_cyan(),
            speedup
        );

        println!("\n{}", "--- CABI Batch Generation Benchmarks ---".bold());
        let mut buffer_u32 = vec![0u32; N];
        let mut buffer_f32 = vec![0f32; N];
        let mut buffer_i32 = vec![0i32; N];

        let p32_ptr = philox32x4_new(12345);
        let p32x4x4_ptr = philox32x4x4_new(12345);

        // u32s
        let start = Instant::now();
        for _ in 0..M {
            philox32x4_next_u32s(p32_ptr, buffer_u32.as_mut_ptr(), N);
        }
        let p32_u32s = start.elapsed() / M;
        println!("{}: {:?}", "Philox32 next_u32s".bright_green(), p32_u32s);

        let start = Instant::now();
        for _ in 0..M {
            philox32x4x4_next_u32s(p32x4x4_ptr, buffer_u32.as_mut_ptr(), N);
        }
        let p32x4x4_u32s = start.elapsed() / M;
        println!(
            "{}: {:?}",
            "Philox32x4x4 next_u32s".bright_green(),
            p32x4x4_u32s
        );
        println!(
            "{}: {:.2}x\n",
            "u32s speedup".bright_cyan(),
            p32_u32s.as_secs_f64() / p32x4x4_u32s.as_secs_f64()
        );

        // f32s
        let start = Instant::now();
        for _ in 0..M {
            philox32x4_next_f32s(p32_ptr, buffer_f32.as_mut_ptr(), N);
        }
        let p32_f32s = start.elapsed() / M;
        println!("{}: {:?}", "Philox32 next_f32s".bright_green(), p32_f32s);

        let start = Instant::now();
        for _ in 0..M {
            philox32x4x4_next_f32s(p32x4x4_ptr, buffer_f32.as_mut_ptr(), N);
        }
        let p32x4x4_f32s = start.elapsed() / M;
        println!(
            "{}: {:?}",
            "Philox32x4x4 next_f32s".bright_green(),
            p32x4x4_f32s
        );
        println!(
            "{}: {:.2}x\n",
            "f32s speedup".bright_cyan(),
            p32_f32s.as_secs_f64() / p32x4x4_f32s.as_secs_f64()
        );

        // i32s bounded
        let start = Instant::now();
        for _ in 0..M {
            philox32x4_rand_i32s(p32_ptr, buffer_i32.as_mut_ptr(), N, -100, 100);
        }
        let p32_i32s = start.elapsed() / M;
        println!("{}: {:?}", "Philox32 rand_i32s".bright_green(), p32_i32s);

        let start = Instant::now();
        for _ in 0..M {
            philox32x4x4_rand_i32s(p32x4x4_ptr, buffer_i32.as_mut_ptr(), N, -100, 100);
        }
        let p32x4x4_i32s = start.elapsed() / M;
        println!(
            "{}: {:?}",
            "Philox32x4x4 rand_i32s".bright_green(),
            p32x4x4_i32s
        );
        println!(
            "{}: {:.2}x\n",
            "i32s speedup".bright_cyan(),
            p32_i32s.as_secs_f64() / p32x4x4_i32s.as_secs_f64()
        );

        // f32s bounded
        let start = Instant::now();
        philox32x4_rand_f32s(p32_ptr, buffer_f32.as_mut_ptr(), N, -1.0, 1.0);
        let p32_randf = start.elapsed();
        println!("{}: {:?}", "Philox32 rand_f32s".bright_green(), p32_randf);

        let start = Instant::now();
        philox32x4x4_rand_f32s(p32x4x4_ptr, buffer_f32.as_mut_ptr(), N, -1.0, 1.0);
        let p32x4x4_randf = start.elapsed();
        println!(
            "{}: {:?}",
            "Philox32x4x4 rand_f32s".bright_green(),
            p32x4x4_randf
        );
        println!(
            "{}: {:.2}x\n",
            "bounded f32s speedup".bright_cyan(),
            p32_randf.as_secs_f64() / p32x4x4_randf.as_secs_f64()
        );

        philox32x4_free(p32_ptr);
        philox32x4x4_free(p32x4x4_ptr);

        let philox32 = philox32_new(0);
        let start = Instant::now();
        philox32_rand_f32s(philox32, buffer_f32.as_mut_ptr(), N, -1.0, 1.0);
        let p32_randf = start.elapsed();
        println!("{}: {:?}", "Philox32 rand_f32s".bright_green(), p32_randf);

        philox32_free(philox32);

        black_box(buffer_f32);
        black_box(buffer_u32);
         black_box(buffer_i32);
    */

    benchmark_throughput_32x4x4(N);
    benchmark_throughput_32x4(N);
    benchmark_throughput_32x4x4_cabi(N);
    benchmark_throughput_32x4_cabi(N);
}

fn benchmark_throughput_32x4x4(n: usize) {
    let mut rng = Philox32x4x4::new(0);
    let d = 16;

    let start = Instant::now();
    for _ in 0..n / (d * 4) {
        black_box(rng.nextu());
        black_box(rng.nextu());
        black_box(rng.nextu());
        black_box(rng.nextu());
    }
    let duration = start.elapsed();

    let throughput = n as f64 / duration.as_secs_f64() / 1_000_000_000.0;
    println!("Philox32x4x4 Pure Throughput: {:.2} Giegel/sec", throughput);
}

fn benchmark_throughput_32x4(n: usize) {
    let mut rng = Philox32x4::new(0);
    let start = Instant::now();

    for _ in 0..n / 4 {
        black_box(rng.nextu());
    }
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / 1_000_000_000.0;
    println!("Philox32x4 Pure Throughput: {:.2} Giegel/sec", throughput);
}

fn benchmark_throughput_32x4x4_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = philox32x4x4_new(0);
    let start = Instant::now();
    philox32x4x4_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / 1_000_000_000.0;
    println!("Philox32x4x4 CABI Throughput: {:.2} Giegel/sec", throughput);
}

fn benchmark_throughput_32x4_cabi(n: usize) {
    let mut buffer = vec![0u32; n];
    let ptr = philox32x4_new(0);
    let start = Instant::now();
    philox32x4_next_u32s(ptr, buffer.as_mut_ptr(), n);
    let duration = start.elapsed();
    let throughput = n as f64 / duration.as_secs_f64() / 1_000_000_000.0;
    println!("Philox32x4 CABI Throughput: {:.2} Giegel/sec", throughput);
}
