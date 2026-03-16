use crate::rng::{Rng32, Rng64};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_rdrand32_step, _rdrand64_step, _rdseed32_step, _rdseed64_step};

#[cfg(target_arch = "x86")]
use std::arch::x86::{_rdrand32_step, _rdseed32_step};

// --- 32-bit constants ---
const SEED_MIX_INCREMENT: u32 = 0x9E3779B9;
const SEED_MIX_MULTIPLIER: u32 = 0x85eb_ca6b;
const FALLBACK_MULTIPLIER: u32 = 0x27d4_eb2d;

// --- 64-bit constants ---
const SEED_MIX_INCREMENT_64: u64 = 0x9E37_79B9_7F4A_7C15;
const SEED_MIX_MULTIPLIER_64: u64 = 0xff51_afd7_ed55_8ccd;
const FALLBACK_MULTIPLIER_64: u64 = 0x2545_f491_4f6c_dd1d;

// ── SeedGen32 ──────────────────────────────────────────────

pub struct SeedGen32<'a, R: Rng32> {
    rng: &'a mut R,
    seed: u32,
}

impl<'a, R: Rng32> SeedGen32<'a, R> {
    pub fn new(rng: &'a mut R, seed: u32) -> Self {
        Self { rng, seed }
    }

    /// Produces the next seed pair derived from hardware noise.
    ///
    /// Returns `(raw, processed)` where `raw` is the value read from the
    /// hardware noise source (RDSEED/RDRAND when available, x86 and x86_64)
    /// and `processed` is the mixed value that updates the internal seed state.
    pub fn next_seed_pair(&mut self) -> (u32, u32) {
        let raw = self.noise();
        let processed = self.process(raw);
        (raw, processed)
    }

    fn process(&mut self, raw: u32) -> u32 {
        let rng_mix = self.rng.randi(0, i32::MAX) as u32;
        let mut value = raw ^ rng_mix;
        value = value.wrapping_add(self.seed);
        value = value.wrapping_add(SEED_MIX_INCREMENT);
        value ^= value >> 16;
        value = value.wrapping_mul(SEED_MIX_MULTIPLIER);
        value ^= value >> 13;
        self.seed = value;
        value
    }

    fn noise(&self) -> u32 {
        hardware_noise32().unwrap_or_else(|| fallback_noise32(self.seed))
    }
}

// ── SeedGen64 ──────────────────────────────────────────────

pub struct SeedGen64<'a, R: Rng64> {
    rng: &'a mut R,
    seed: u64,
}

impl<'a, R: Rng64> SeedGen64<'a, R> {
    pub fn new(rng: &'a mut R, seed: u64) -> Self {
        Self { rng, seed }
    }

    /// Produces the next seed pair derived from hardware noise.
    ///
    /// Returns `(raw, processed)` where `raw` is the value read from the
    /// hardware noise source (RDSEED64/RDRAND64 on x86_64; two RDSEED32 calls
    /// combined on x86) and `processed` is the mixed value that updates the
    /// internal seed state.
    pub fn next_seed_pair(&mut self) -> (u64, u64) {
        let raw = self.noise();
        let processed = self.process(raw);
        (raw, processed)
    }

    fn process(&mut self, raw: u64) -> u64 {
        let rng_mix = self.rng.randi(0, i64::MAX) as u64;
        let mut value = raw ^ rng_mix;
        value = value.wrapping_add(self.seed);
        value = value.wrapping_add(SEED_MIX_INCREMENT_64);
        value ^= value >> 33;
        value = value.wrapping_mul(SEED_MIX_MULTIPLIER_64);
        value ^= value >> 29;
        self.seed = value;
        value
    }

    fn noise(&self) -> u64 {
        hardware_noise64().unwrap_or_else(|| fallback_noise64(self.seed))
    }
}

// ── hardware noise ─────────────────────────────────────────

fn hardware_noise32() -> Option<u32> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("rdseed") {
            unsafe {
                if let Some(v) = rdseed32_once() {
                    return Some(v);
                }
            }
        }
        if std::arch::is_x86_feature_detected!("rdrand") {
            unsafe {
                if let Some(v) = rdrand32_once() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn hardware_noise64() -> Option<u64> {
    // x86_64: native 64-bit instructions
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("rdseed") {
            unsafe {
                if let Some(v) = rdseed64_once() {
                    return Some(v);
                }
            }
        }
        if std::arch::is_x86_feature_detected!("rdrand") {
            unsafe {
                if let Some(v) = rdrand64_once() {
                    return Some(v);
                }
            }
        }
    }

    // x86 (32-bit): combine two 32-bit samples
    #[cfg(target_arch = "x86")]
    if let (Some(lo), Some(hi)) = (hardware_noise32(), hardware_noise32()) {
        return Some((hi as u64) << 32 | lo as u64);
    }

    None
}

// ── fallback noise ─────────────────────────────────────────

fn fallback_noise32(seed: u32) -> u32 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut value = (now as u32)
        .wrapping_add(seed.rotate_left(7))
        .wrapping_mul(FALLBACK_MULTIPLIER);
    value ^= ((now >> 32) as u32).wrapping_add(seed.rotate_right(5));
    value ^ (value >> 15)
}

fn fallback_noise64(seed: u64) -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut value = (now as u64)
        .wrapping_add(seed.rotate_left(11))
        .wrapping_mul(FALLBACK_MULTIPLIER_64);
    value ^= ((now >> 64) as u64).wrapping_add(seed.rotate_right(7));
    value ^ (value >> 31)
}

// ── x86 / x86_64 intrinsics ───────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn rdseed32_once() -> Option<u32> {
    let mut value = 0u32;
    for _ in 0..4 {
        if unsafe { _rdseed32_step(&mut value) } == 1 {
            return Some(value);
        }
    }
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn rdrand32_once() -> Option<u32> {
    let mut value = 0u32;
    for _ in 0..4 {
        if unsafe { _rdrand32_step(&mut value) } == 1 {
            return Some(value);
        }
    }
    None
}

#[cfg(target_arch = "x86_64")]
unsafe fn rdseed64_once() -> Option<u64> {
    let mut value = 0u64;
    for _ in 0..4 {
        if unsafe { _rdseed64_step(&mut value) } == 1 {
            return Some(value);
        }
    }
    None
}

#[cfg(target_arch = "x86_64")]
unsafe fn rdrand64_once() -> Option<u64> {
    let mut value = 0u64;
    for _ in 0..4 {
        if unsafe { _rdrand64_step(&mut value) } == 1 {
            return Some(value);
        }
    }
    None
}
