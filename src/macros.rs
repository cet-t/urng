//! Macros for quick random number generation.
//!
//! These macros provide a convenient way to generate random numbers using various
//! algorithms with seeds generated from the current system time.

#[doc(hidden)]
/// Internal helper to generate a seed from the current system time.
pub fn __get_seed() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(std::time::Duration::from_secs(0))
        .as_nanos()
}

#[macro_export]
/// Generates the next random value using the specified algorithm and output type.
///
/// Seeds are automatically generated from the system time.
///
/// The argument format is `[algorithm][bits][type]`.
/// - `type`: `u` for unsigned integer, `f` for floating-point [0, 1).
///
/// # Examples
/// ```
/// use urng::next;
///
/// let val_u32 = next!(mt32u);
/// let val_f32 = next!(xor32f);
/// ```
macro_rules! next {
    // --- 32-bit output variants ---
    (xor32u) => {
        $crate::rng32::Xorshift32::new($crate::macros::__get_seed() as u32).nextu()
    };
    (xor32f) => {
        $crate::rng32::Xorshift32::new($crate::macros::__get_seed() as u32).nextf()
    };
    (mt32u) => {
        $crate::rng32::Mt19937::new($crate::macros::__get_seed() as u32).nextu()
    };
    (mt32f) => {
        $crate::rng32::Mt19937::new($crate::macros::__get_seed() as u32).nextf()
    };
    (pcg32u) => {
        $crate::rng32::Pcg32::new($crate::macros::__get_seed() as u64).nextu()
    };
    (pcg32f) => {
        $crate::rng32::Pcg32::new($crate::macros::__get_seed() as u64).nextf()
    };
    (philox32u) => {
        $crate::rng32::Philox32::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
        ])
        .nextu()
    };
    (philox32f) => {
        $crate::rng32::Philox32::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
        ])
        .nextf()
    };
    (tgfsr32u) => {
        $crate::rng32::TwistedGFSR::new($crate::rng32::TwistedGFSR::new_seed()).nextu()
    }; // TGFSR uses fixed seed array or needs complex initialization
    (tgfsr32f) => {
        $crate::rng32::TwistedGFSR::new($crate::rng32::TwistedGFSR::new_seed()).nextf()
    };

    // --- 64-bit output variants ---
    (xor64u) => {
        $crate::rng64::Xorshift64::new($crate::macros::__get_seed() as u64).nextu()
    };
    (xor64f) => {
        $crate::rng64::Xorshift64::new($crate::macros::__get_seed() as u64).nextf()
    };
    (mt64u) => {
        $crate::rng64::Mt1993764::new($crate::macros::__get_seed() as u64, 3).nextu()
    };
    (mt64f) => {
        $crate::rng64::Mt1993764::new($crate::macros::__get_seed() as u64, 3).nextf()
    };
    (philox64u) => {
        $crate::rng64::Philox64::new([
            $crate::macros::__get_seed() as u64,
            ($crate::macros::__get_seed() >> 64) as u64,
        ])
        .nextu()
    };
    (philox64f) => {
        $crate::rng64::Philox64::new([
            $crate::macros::__get_seed() as u64,
            ($crate::macros::__get_seed() >> 64) as u64,
        ])
        .nextf()
    };

    // --- 128-bit internal state (32-bit output) ---
    (xor128u) => {
        $crate::rng128::Xorshift128::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
            ($crate::macros::__get_seed() >> 64) as u32,
            ($crate::macros::__get_seed() >> 96) as u32,
        ])
        .nextu()
    };
    (xor128f) => {
        $crate::rng128::Xorshift128::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
            ($crate::macros::__get_seed() >> 64) as u32,
            ($crate::macros::__get_seed() >> 96) as u32,
        ])
        .nextf()
    };
}

#[macro_export]
/// Generates a random value in the specified range using the specified algorithm and output type.
///
/// Seeds are automatically generated from the system time.
///
/// The argument format is `[algorithm][bits][type]`.
/// - `type`: `i` for signed integer [min, max], `f` for floating-point [min, max).
///
/// # Examples
/// ```
/// use urng::rand;
///
/// let val_i32 = rand!(xor32i; 1, 100);
/// let val_f64 = rand!(mt64f; 0.5, 1.5);
/// ```
macro_rules! rand {
    // --- 32-bit output variants ---
    (xor32i; $min:expr, $max:expr) => {
        $crate::rng32::Xorshift32::new($crate::macros::__get_seed() as u32).randi($min, $max)
    };
    (xor32f; $min:expr, $max:expr) => {
        $crate::rng32::Xorshift32::new($crate::macros::__get_seed() as u32).randf($min, $max)
    };
    (mt32i; $min:expr, $max:expr) => {
        $crate::rng32::Mt19937::new($crate::macros::__get_seed() as u32, 3).randi($min, $max)
    };
    (mt32f; $min:expr, $max:expr) => {
        $crate::rng32::Mt19937::new($crate::macros::__get_seed() as u32, 3).randf($min, $max)
    };
    (pcg32i; $min:expr, $max:expr) => {
        $crate::rng32::Pcg32::new($crate::macros::__get_seed() as u64).randi($min, $max)
    };
    (pcg32f; $min:expr, $max:expr) => {
        $crate::rng32::Pcg32::new($crate::macros::__get_seed() as u64).randf($min, $max)
    };
    (philox32i; $min:expr, $max:expr) => {
        $crate::rng32::Philox32::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
        ])
        .randi($min, $max)
    };
    (philox32f; $min:expr, $max:expr) => {
        $crate::rng32::Philox32::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
        ])
        .randf($min, $max)
    };
    (tgfsr32i; $min:expr, $max:expr) => {
        $crate::rng32::TwistedGFSR::new($crate::rng32::TwistedGFSR::new_seed()).randi($min, $max)
    };
    (tgfsr32f; $min:expr, $max:expr) => {
        $crate::rng32::TwistedGFSR::new($crate::rng32::TwistedGFSR::new_seed()).randf($min, $max)
    };

    // --- 64-bit output variants ---
    (xor64i; $min:expr, $max:expr) => {
        $crate::rng64::Xorshift64::new($crate::macros::__get_seed() as u64).randi($min, $max)
    };
    (xor64f; $min:expr, $max:expr) => {
        $crate::rng64::Xorshift64::new($crate::macros::__get_seed() as u64).randf($min, $max)
    };
    (mt64i; $min:expr, $max:expr) => {
        $crate::rng64::Mt1993764::new($crate::macros::__get_seed() as u64, 3).randi($min, $max)
    };
    (mt64f; $min:expr, $max:expr) => {
        $crate::rng64::Mt1993764::new($crate::macros::__get_seed() as u64, 3).randf($min, $max)
    };
    (philox64i; $min:expr, $max:expr) => {
        $crate::rng64::Philox64::new([
            $crate::macros::__get_seed() as u64,
            ($crate::macros::__get_seed() >> 64) as u64,
        ])
        .randi($min, $max)
    };
    (philox64f; $min:expr, $max:expr) => {
        $crate::rng64::Philox64::new([
            $crate::macros::__get_seed() as u64,
            ($crate::macros::__get_seed() >> 64) as u64,
        ])
        .randf($min, $max)
    };

    // --- 128-bit internal state (32-bit output) ---
    (xor128i; $min:expr, $max:expr) => {
        $crate::rng128::Xorshift128::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
            ($crate::macros::__get_seed() >> 64) as u32,
            ($crate::macros::__get_seed() >> 96) as u32,
        ])
        .randi($min, $max)
    };
    (xor128f; $min:expr, $max:expr) => {
        $crate::rng128::Xorshift128::new([
            $crate::macros::__get_seed() as u32,
            ($crate::macros::__get_seed() >> 32) as u32,
            ($crate::macros::__get_seed() >> 64) as u32,
            ($crate::macros::__get_seed() >> 96) as u32,
        ])
        .randf($min, $max)
    };
}

#[macro_export]
/// Wraps a value in a `Wrapping` type.
///
/// Can also be used to create arrays of `Wrapping` values.
///
/// # Examples
/// ```
/// use urng::wrap;
///
/// // Single value
/// let val = wrap!(1);
/// assert_eq!(val, std::num::Wrapping(1));
///
/// // Array with repeated value
/// let arr = wrap![1; 3];
/// assert_eq!(arr, [std::num::Wrapping(1), std::num::Wrapping(1), std::num::Wrapping(1)]);
///
/// // Array with specific values
/// let arr2 = wrap![1, 2, 3];
/// assert_eq!(arr2, [std::num::Wrapping(1), std::num::Wrapping(2), std::num::Wrapping(3)]);
/// ```
macro_rules! wrap {
    ($a:expr) => {
        ::std::num::Wrapping($a)
    };

    ($elem:expr; $n:expr) => (
        [::std::num::Wrapping($elem); $n]
    );

    ($($x:expr),+ $(,)?) => (
        [$(::std::num::Wrapping($x)),+]
    );

}

#[macro_export]
/// Randomly selects an index based on weights using a binary search approach.
///
/// # Examples
/// ```
/// use urng::search;
///
/// let mut rng = urng::rng64::Mt1993764::new(1, 256);
/// let index = search!(&mut rng, [1.0, 9.0]);
/// assert!(index == Some(0) || index == Some(1));
/// ```
macro_rules! search {
    ($rng:expr, $weights:expr) => {
        $crate::bst::search($rng, &$weights)
    };
}

#[macro_export]
macro_rules! choice {
    ($rng:expr, $weights:expr, $items:expr) => {
        $crate::bst::choice($rng, &$weights, &$items)
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        // Just verify they don't panic and return values
        let _ = next!(xor32u);
        let _ = next!(mt32f);
        let _ = next!(xor64u);
        let _ = next!(xor128u);

        let val = rand!(xor32i; 1, 10);
        assert!(val >= 1 && val <= 10);

        let fval = rand!(xor32f; 0.0, 1.0);
        assert!(fval >= 0.0 && fval < 1.0);

        let val64 = rand!(mt64i; 100, 200);
        assert!(val64 >= 100 && val64 <= 200);
    }
}
