use crate::_internal::{i2f_bits, randi_wide, u2f_01};

mod sealed {
    /// Prevents `Word` from being implemented outside this crate, so the
    /// `u32 -> f32/i32` / `u64 -> f64/i64` width relationships stay closed.
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// The unsigned output word of a generator (`u32` or `u64`).
///
/// A `Word` fixes the whole numeric family of an RNG — its float type, its
/// signed-range integer type, and the bit width — and centralizes the
/// uniform-conversion formulas in one place so every [`Rng`] implementor
/// inherits them for free.
pub trait Word: sealed::Sealed + Copy {
    /// The float type produced from this word (`f32` for `u32`, `f64` for `u64`).
    type Float: Copy;
    /// The signed integer type used for ranged draws (`i32` / `i64`).
    type Int: Copy;
    /// The bit width of the word (`32` or `64`).
    const BITS: u32;

    /// Maps the word uniformly onto `[0, 1)`.
    fn to_f01(self) -> Self::Float;
    /// Maps the word uniformly onto the inclusive integer range `[min, max]`.
    fn to_randi(self, min: Self::Int, max: Self::Int) -> Self::Int;
    /// Maps the word uniformly onto the half-open float range `[min, max)`.
    fn to_randf(self, min: Self::Float, max: Self::Float) -> Self::Float;
    /// Maps the word uniformly onto a slice index in `[0, len)` (Lemire's
    /// multiply-shift). Equivalent to `to_randi(0, len as Int - 1) as usize`.
    fn to_index(self, len: usize) -> usize;
}

macro_rules! impl_word {
    ($bits:tt) => {
        pastey::paste! {
            impl Word for [<u $bits>] {
                type Float = [<f $bits>];
                type Int = [<i $bits>];
                const BITS: u32 = $bits;

                #[inline(always)]
                fn to_f01(self) -> Self::Float {
                    u2f_01!([<f $bits>], $bits, self)
                }

                #[inline(always)]
                fn to_randi(self, min: Self::Int, max: Self::Int) -> Self::Int {
                    let range = (max as randi_wide!(i $bits) - min as randi_wide!(i $bits) + 1)
                        as randi_wide!(u $bits);
                    ((self as randi_wide!(u $bits) * range) >> $bits) as [<i $bits>] + min
                }

                #[inline(always)]
                fn to_randf(self, min: Self::Float, max: Self::Float) -> Self::Float {
                    self.to_f01() * (max - min) + min
                }

                #[inline(always)]
                fn to_index(self, len: usize) -> usize {
                    ((self as randi_wide!(u $bits) * len as randi_wide!(u $bits)) >> $bits) as usize
                }
            }
        }
    };
}

impl_word!(32);
impl_word!(64);

/// A random number generator characterized by a single output [`Word`].
///
/// Implementors provide only [`nextu`](Rng::nextu); the ranged and float
/// draws are supplied as defaults derived from the associated [`Word`], so the
/// numeric formulas live in exactly one place.
pub trait Rng {
    /// The unsigned output word this generator produces.
    type Word: Word;

    /// Generates the next raw word in `[0, 2^BITS)`.
    fn nextu(&mut self) -> Self::Word;

    /// Generates the next float in `[0, 1)`.
    #[inline(always)]
    fn nextf(&mut self) -> <Self::Word as Word>::Float {
        self.nextu().to_f01()
    }

    /// Generates an integer uniformly in the inclusive range `[min, max]`.
    #[inline(always)]
    fn randi(
        &mut self,
        min: <Self::Word as Word>::Int,
        max: <Self::Word as Word>::Int,
    ) -> <Self::Word as Word>::Int {
        self.nextu().to_randi(min, max)
    }

    /// Generates a float uniformly in the half-open range `[min, max)`.
    #[inline(always)]
    fn randf(
        &mut self,
        min: <Self::Word as Word>::Float,
        max: <Self::Word as Word>::Float,
    ) -> <Self::Word as Word>::Float {
        self.nextu().to_randf(min, max)
    }
}

/// Constructs a generator from a raw seed of its native word width.
pub trait Seed {
    /// The seed type (`u32` or `u64`, matching the generator's word width).
    type Seed;
    /// Builds `Self` from a seed.
    fn from_seed(seed: Self::Seed) -> Self;
}
