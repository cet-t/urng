use urng::{rng32::*, rng64::*};

const N: usize = 10; //i32::MAX as usize;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("N: {}", N);

    let rng = xorshift32_new(0x00000001);
    let mut x = [0.0; N];
    let mut y = [0.0; N];
    xorshift32_rand_f32s(rng, x.as_mut_ptr(), N, -9.0, 0.0);
    xorshift32_rand_f32s(rng, y.as_mut_ptr(), N, 0.0, 5.0);
    for i in 0..N {
        println!("{}, {}", x[i], y[i]);
    }

    Ok(())
}
