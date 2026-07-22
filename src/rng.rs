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

macro_rules! u2f_01 {
    ($ft:ty, $bits:tt, $x:expr) => {{
        <$ft>::from_bits(($x >> i2f_bits!($bits bias)) | i2f_bits!($bits bits)) - 1.0
    }};
}

macro_rules! impl_rng_trait {
    ($bits:expr) => {
        pastey::paste! {
            pub trait [<Seed $bits>] {
                fn from_seed(seed: [<u $bits>]) -> Self;
            }

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
                    u2f_01!([<f $bits>], $bits, self.nextu())
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
                    let base = u2f_01!([<f $bits>], $bits, self.nextu());
                    base * (max - min) + min
                }
            }

            $crate::impl_choice!($bits);
            $crate::impl_shuffle!($bits);
        }
    };
}

impl_rng_trait!(32);
impl_rng_trait!(64);
