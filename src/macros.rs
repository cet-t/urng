//! Macros for quick random number generation.
//!
//! These macros provide a convenient way to generate random numbers using various
//! algorithms with seeds generated from the current system time.

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
/// Dispatches to an AVX-512 optimized path on x86_64 when available, otherwise falls back.
///
/// Two forms:
/// - `dispatch_simd!(RetType, fallback_fn, avx512_fn, seed)` — allocate and return a raw pointer.
/// - `dispatch_simd!(Avx512T, FallbackT, fallback_fn, avx512_fn, ptr [, args])` — operate in-place.
macro_rules! dispatch_simd {
    ($ret_type:ty, $fallback_fn:ident, $avx512_fn:ident, $seed:expr) => {{
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            return $avx512_fn($seed) as *mut $ret_type;
        }
        $fallback_fn($seed) as *mut $ret_type
    }};
    ($avx512_type:ty, $fallback_type:ty, $fallback_fn:ident, $avx512_fn:ident, $ptr:expr $(, $arg:expr)*) => {{
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx512f") {
            $avx512_fn($ptr as *mut $avx512_type $(, $arg)*);
            return;
        }
        $fallback_fn($ptr as *mut $fallback_type $(, $arg)*);
    }};
}
