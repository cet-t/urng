//! Hardware-noise-based seed generator.
//!
//! [`SeedGen`] mixes hardware noise (RDSEED/RDRAND on x86/x86_64) with an
//! existing RNG to produce high-quality seed values, for either `u32` or
//! `u64` word width. Falls back to a timestamp-based noise source on
//! platforms without those instructions.
//!
//! Enabled by the `seedgen` crate feature.
//!
//! # Examples
//!
//! ```
//! use urng::SeedGen;
//! use urng::SplitMix32;
//!
//! let mut rng = SplitMix32::new(0);
//! let mut sg = SeedGen::new(&mut rng, 0u32);
//! let (raw, processed): (u32, u32) = sg.next_seed_pair();
//! assert_eq!(raw <= u32::MAX, true);
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use wrapn::wrap;

use crate::rng::{Rng, Word};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_rdrand32_step, _rdrand64_step, _rdseed32_step, _rdseed64_step};

#[cfg(target_arch = "x86")]
use std::arch::x86::{_rdrand32_step, _rdseed32_step};

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// A [`Word`] that can drive [`SeedGen`] (`u32` or `u64`).
///
/// Centralizes the hardware-noise source, timestamp fallback, and Murmur3-style
/// mixing formula for each word width, so [`SeedGen`] itself stays width-agnostic.
pub trait SeedWord: sealed::Sealed + Word {
    /// Reads hardware noise (RDSEED/RDRAND) if available on this platform.
    fn hardware_noise() -> Option<Self>;
    /// Timestamp-based fallback noise source, used when hardware noise is unavailable.
    fn fallback_noise(seed: Self) -> Self;
    /// Draws a mixing value from `rng`, in this word's native width.
    fn rng_mix<R: Rng<Word = Self>>(rng: &mut R) -> Self;
    /// Mixes `raw` noise with `rng_mix` and the running `seed` into the next seed value.
    fn mix(raw: Self, rng_mix: Self, seed: Self) -> Self;
}

macro_rules! impl_seed_word {
    ($bits:expr, $inc:expr, $mult:expr, $offsets:expr) => {
        ::pastey::paste! {
            impl SeedWord for [<u $bits>] {
                fn hardware_noise() -> Option<Self> {
                    [<hardware_noise $bits>]()
                }

                fn fallback_noise(seed: Self) -> Self {
                    [<fallback_noise $bits>](seed)
                }

                fn rng_mix<R: Rng<Word = Self>>(rng: &mut R) -> Self {
                    rng.randi(0, [<i $bits>]::MAX) as [<u $bits>]
                }

                fn mix(raw: Self, rng_mix: Self, seed: Self) -> Self {
                    let mut value = wrap!(raw ^ rng_mix);
                    value += seed;
                    value += $inc;
                    value ^= value >> $offsets[0];
                    value *= $mult;
                    value ^= value >> $offsets[1];
                    *value
                }
            }
        }
    };
}

impl_seed_word!(32, 0x9E3779B9, 0x85eb_ca6b, [16, 13]);
impl_seed_word!(64, 0x9E37_79B9_7F4A_7C15, 0xff51_afd7_ed55_8ccd, [33, 29]);

/// Hardware-noise-assisted seed generator.
///
/// Wraps an existing [`Rng`] and mixes hardware noise (RDSEED/RDRAND on x86/x86_64,
/// timestamp fallback elsewhere) into a Murmur3-style hash to produce reproducibly
/// high-entropy seed values, in the RNG's own word width (`u32` or `u64`).
///
/// # Examples
///
/// ```
/// use urng::SeedGen;
/// use urng::SplitMix64;
///
/// let mut rng = SplitMix64::new(12345);
/// let mut sg = SeedGen::new(&mut rng, 0u64);
/// let (raw, seed): (u64, u64) = sg.next_seed_pair();
/// assert_eq!(raw <= u64::MAX, true);
/// ```
pub struct SeedGen<'a, R: Rng>
where
    R::Word: SeedWord,
{
    rng: &'a mut R,
    seed: R::Word,
}

impl<'a, R: Rng> SeedGen<'a, R>
where
    R::Word: SeedWord,
{
    /// Creates a new `SeedGen` wrapping `rng` with the given initial `seed`.
    pub fn new(rng: &'a mut R, seed: R::Word) -> Self {
        Self { rng, seed }
    }

    /// Produces the next seed pair derived from hardware noise.
    ///
    /// Returns `(raw, processed)` where `raw` is the value read from the
    /// hardware noise source (RDSEED/RDRAND when available) and `processed`
    /// is the mixed value that updates the internal seed state.
    pub fn next_seed_pair(&mut self) -> (R::Word, R::Word) {
        let raw = self.noise();
        let processed = self.process(raw);
        (raw, processed)
    }

    fn process(&mut self, raw: R::Word) -> R::Word {
        let rng_mix = R::Word::rng_mix(self.rng);
        let value = R::Word::mix(raw, rng_mix, self.seed);
        self.seed = value;
        value
    }

    fn noise(&self) -> R::Word {
        R::Word::hardware_noise().unwrap_or_else(|| R::Word::fallback_noise(self.seed))
    }
}

fn hardware_noise32() -> Option<u32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("rdseed")
            && let Some(v) = unsafe { rdseed32_once() }
        {
            return Some(v);
        }
        if std::arch::is_x86_feature_detected!("rdrand")
            && let Some(v) = unsafe { rdrand32_once() }
        {
            return Some(v);
        }
    }
    None
}

fn hardware_noise64() -> Option<u64> {
    // x86_64: native 64-bit instructions
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("rdseed")
            && let Some(v) = unsafe { rdseed64_once() }
        {
            return Some(v);
        }
        if std::arch::is_x86_feature_detected!("rdrand")
            && let Some(v) = unsafe { rdrand64_once() }
        {
            return Some(v);
        }
    }

    // x86 (32-bit): combine two 32-bit samples
    #[cfg(target_arch = "x86")]
    if let (Some(lo), Some(hi)) = (hardware_noise32(), hardware_noise32()) {
        return Some((hi as u64) << 32 | lo as u64);
    }

    None
}

macro_rules! impl_noise {
    ($bits:expr, $offsets:expr, $mult:expr) => {
        ::pastey::paste! {
            fn [<fallback_noise $bits>](seed: [<u $bits>]) -> [<u $bits>] {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();

                let mut value = (now as [<u $bits>])
                    .wrapping_add(seed.rotate_left($offsets[0]))
                    .wrapping_mul($mult);
                value ^= ((now >> $bits) as [<u $bits>]).wrapping_add(seed.rotate_right($offsets[1]));
                value ^ (value >> ($bits / 2 - 1))
            }
        }
    };
}

impl_noise!(32, [7, 5], 0x27d4eb2d);
impl_noise!(64, [11, 7], 0x2545_f491_4f6c_dd1d);

macro_rules! impl_rd {
    ($bits:expr, $($arches:expr),+) => {
        ::pastey::paste! {
            #[cfg(any($(target_arch = $arches),+))]
            unsafe fn [<rdseed $bits _once>]() -> Option<[<u $bits>]> {
                let mut value = 0 as [<u $bits>];
                for _ in 0..4 {
                    if unsafe { [<_rdseed $bits _step>](&mut value) } == 1 {
                        return Some(value);
                    }
                }
                None
            }

            #[cfg(any($(target_arch = $arches),+))]
            unsafe fn [<rdrand $bits _once>]() -> Option<[<u $bits>]> {
                let mut value = 0 as [<u $bits>];
                for _ in 0..4 {
                    if unsafe { [<_rdrand $bits _step>](&mut value) } == 1 {
                        return Some(value);
                    }
                }
                None
            }
        }
    };
}

impl_rd!(32, "x86", "x86_64");
impl_rd!(64, "x86_64");
