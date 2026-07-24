#[macro_export]
macro_rules! i2f_bits {
    (32 bits) => {
        0x3F800000
    };
    (32 bias) => {
        9
    };
    (64 bits) => {
        0x3FF0000000000000
    };
    (64 bias) => {
        11
    };
}

/// Generates the common convenience methods (`nextf`, `randi`, `randf`) for a
/// `wide` SIMD generator.
///
/// `$size` is the number of SIMD lanes and `$bits` is the integer width (32 or 64).
/// The generated functions mirror the scalar [`crate::rng::Rng`]/`Rng` helpers
/// but return arrays of `$size` values produced in parallel.
macro_rules! impl_methods {
    ($size:expr, $bits:tt) => {
        ::pastey::paste! {
            #[doc = "Generates random `f32` values uniformly distributed in `[0, 1)`."]
            #[doc = ""]
            #[doc = "The raw integer output is biased into the exponent and mantissa bits of an"]
            #[doc = "`f32` (yielding a value in `[1, 2)`) and then `1.0` is subtracted."]
            #[inline(always)]
            pub fn nextf(&mut self) -> [[<f $bits>]; $size] {
                self.nextu()
                    .map(|x| [<f $bits>]::from_bits((x >> $crate::i2f_bits!($bits bias)) | $crate::i2f_bits!($bits bits)) - 1.0)
            }

            #[doc = "Generates random `i32` values uniformly distributed in `[min, max]` (inclusive)."]
            #[doc = ""]
            #[doc = "Uses Lemire's nearly-divisionless reduction over the full `[0, 2^32)` output."]
            #[inline(always)]
            pub fn randi(&mut self, min: [<i $bits>], max: [<i $bits>]) -> [[<i $bits>]; $size] {
                let range = (max as i128 - min as i128 + 1) as u128;
                self.nextu().map(|x| (((x as u128) * range >> $bits) as [<i $bits>]) + min)
            }

            #[doc = "Generates random `f32` values uniformly distributed in `[min, max)`."]
            #[doc = ""]
            #[doc = "Scales a `[0, 1)` float by `(max - min)` and offsets it by `min`."]
            #[inline(always)]
            pub fn randf(&mut self, min: [<f $bits>], max: [<f $bits>]) -> [[<f $bits>]; $size] {
                self.nextu().map(|x| {
                    let base = [<f $bits>]::from_bits(
                        (x >> $crate::i2f_bits!($bits bias)) | $crate::i2f_bits!($bits bits),
                    ) - 1.0;
                    base * (max - min) + min
                })
            }
        }
    };
}

macro_rules! wide_rotate_left {
    (32 $x:expr, $shift:expr) => {
        ($x << $shift) | ($x >> (32 - $shift))
    };
    (64 $x:expr, $shift:expr) => {
        ($x << $shift) | ($x >> (64 - $shift))
    };
}

macro_rules! wide_rotate_right {
    (32 $x:expr, $shift:expr) => {
        ($x >> $shift) | ($x << (32 - $shift))
    };
    (64 $x:expr, $shift:expr) => {
        ($x >> $shift) | ($x << (64 - $shift))
    };
}

pub(crate) use impl_methods;
pub(crate) use wide_rotate_left;
pub(crate) use wide_rotate_right;
