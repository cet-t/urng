use std::time::Instant;
use urng::rng::Rng32;
use urng::rng32::{
    Jsf32, Jsf32x16, Pcg32, Sfc32, Sfc32x4, Sfc32x8, Sfc32x16, SplitMix32,
    Xoroshiro64Ssx16, Xoshiro128Pp, Xoshiro128Ppx16,
};

const ITERS: u64 = 1_000_000_000;

macro_rules! bench_scalar {
    ($label:expr, $ctor:expr) => {{
        let mut rng = $ctor;
        let mut sink: u64 = 0;
        let t = Instant::now();
        for _ in 0..ITERS {
            sink ^= rng.nextu() as u64;
        }
        let ns = t.elapsed().as_nanos() as f64;
        let gs = ITERS as f64 / ns;
        println!("{:<22}: {:.3} Gsamples/s  (sink={:#018x})", $label, gs, sink);
    }};
}

macro_rules! bench_arr {
    ($label:expr, $ctor:expr, $n:expr) => {{
        let mut rng = $ctor;
        let mut sink: u64 = 0;
        let calls = ITERS / $n;
        let t = Instant::now();
        for _ in 0..calls {
            for x in rng.nextu() {
                sink ^= x as u64;
            }
        }
        let ns = t.elapsed().as_nanos() as f64;
        let gs = (calls * $n) as f64 / ns;
        println!("{:<22}: {:.3} Gsamples/s  (sink={:#018x})", $label, gs, sink);
    }};
}

fn main() {
    println!("Scalar PRNGs (u32 output)");
    bench_scalar!("SplitMix32",    SplitMix32::new(1));
    bench_scalar!("Sfc32",         Sfc32::new(1));
    bench_scalar!("Pcg32",         Pcg32::new(1u64));
    bench_scalar!("Jsf32",         Jsf32::new(1));
    bench_scalar!("Xoshiro128++",  Xoshiro128Pp::new(1));

    println!("\nVectorized PRNGs (u32xN output)");
    // Sfc32x4: new=safe, nextu=safe
    bench_arr!("Sfc32x4",          Sfc32x4::new(1),                4u64);
    // Sfc32x8/x16: new=unsafe, nextu=safe (transmute)
    bench_arr!("Sfc32x8",          unsafe { Sfc32x8::new(1) },     8u64);
    bench_arr!("Sfc32x16",         unsafe { Sfc32x16::new(1) },    16u64);
    // Jsf32x16: new=safe (target_feature), nextu=safe (transmute)
    bench_arr!("Jsf32x16",         unsafe { Jsf32x16::new(1) },        16u64);
    bench_arr!("Xoroshiro64Ssx16", unsafe { Xoroshiro64Ssx16::new(1) }, 16u64);

    // Xoshiro128Ppx16: new=unsafe, nextu=unsafe
    {
        let mut rng = unsafe { Xoshiro128Ppx16::new(1) };
        let mut sink: u64 = 0;
        let calls = ITERS / 16;
        let t = Instant::now();
        for _ in 0..calls {
            for x in unsafe { rng.nextu() } {
                sink ^= x as u64;
            }
        }
        let ns = t.elapsed().as_nanos() as f64;
        let gs = (calls * 16) as f64 / ns;
        println!("{:<22}: {:.3} Gsamples/s  (sink={:#018x})", "Xoshiro128++x16", gs, sink);
    }
}
