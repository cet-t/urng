use crate::_internal::{FSCALE32, FSCALE64};

macro_rules! randi_wide {
    (i 32) => {
        i64
    };
    (i 64) => {
        i128
    };
    (u 32) => {
        u64
    };
    (u 64) => {
        u128
    };
}

macro_rules! impl_rng_trait {
    ($bits:expr) => {
        paste::paste! {
            #[doc = concat!("A trait for ", $bits, "-bit random number generators.")]
            ///
            /// # Examples
            ///
            /// ```
            #[doc = concat!("use urng::rng::Rng", $bits, ";")]
            #[doc = concat!("use urng::rng::Choice", $bits, ";")]
            #[doc = concat!("use urng::rng", $bits, "::Xorshift", $bits, ";")]
            ///
            #[doc = concat!("let mut rng = Xorshift", $bits, "::new(1);")]
            #[doc = concat!("let val = rng.randi(1, 6);")]
            #[doc = concat!("assert!((1..=6).contains(&val));")]
            #[doc = concat!("assert!(rng.randf(0.0, 1.0_f", $bits, ") < 1.0);")]
            #[doc = concat!("let items = [\"a\", \"b\", \"c\"];")]
            #[doc = concat!("assert!(items.contains(rng.choice(&items)));")]
            /// ```
            pub trait [<Rng $bits>] {
                #[doc = concat!("Generates the next random `u", $bits, "` value in the range [0, 2^", $bits, ").")]
                fn nextu(&mut self) -> [<u $bits>];

                #[doc = concat!("Generates the next random `f", $bits, "` value in the range [0, 1).")]
                #[inline(always)]
                fn nextf(&mut self) -> [<f $bits>] {
                    self.nextu() as [<f $bits>] * [<FSCALE $bits>]
                }

                #[doc = concat!("Generates a random `i", $bits, "` value in the range [min, max].")]
                #[inline(always)]
                fn randi(&mut self, min: [<i $bits>], max: [<i $bits>]) -> [<i $bits>] {
                    let range = (max as randi_wide!(i $bits) - min as randi_wide!(i $bits) + 1)
                        as randi_wide!(u $bits);
                    ((self.nextu() as randi_wide!(u $bits) * range) >> $bits) as [<i $bits>] + min
                }

                #[doc = concat!("Generates a random `f", $bits, "` value in the range [min, max).")]
                #[inline(always)]
                fn randf(&mut self, min: [<f $bits>], max: [<f $bits>]) -> [<f $bits>] {
                    let scale = (max - min) * [<FSCALE $bits>];
                    (self.nextu() as [<f $bits>] * scale) + min
                }
            }

            pub trait [<Choice $bits>]: [<Rng $bits>] {
                /// Returns a random element from a slice.
                #[inline(always)]
                fn choice<'a, T>(&mut self, choices: &'a [T]) -> &'a T {
                    let index = self.randi(0, choices.len() as [<i $bits>] - 1);
                    &choices[index as usize]
                }
            }

            impl<T: [<Rng $bits>] + ?Sized> [<Choice $bits>] for T {}
        }
    };
}

impl_rng_trait!(32);
impl_rng_trait!(64);
