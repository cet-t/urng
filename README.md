[<img alt="crates.io" src="https://img.shields.io/crates/v/urng.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/urng)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-urng-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="20">](https://docs.rs/urng)
[<img alt="github" src="https://img.shields.io/badge/github-cet--t%2Furng-5f5fff?style=for-the-badge&labelColor=555555&logo=github" height="20">](https://github.com/cet-t/urng)

# Universal RNG

A collection of efficient pseudo-random number generators (PRNGs) implemented in pure Rust.
This crate provides a wide variety of algorithms, ranging from standard Mersenne Twister to modern high-performance generators like Xoshiro and Philox.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
urng = "0.8.0"
```

### Optional `rand` Feature

Enable the `rand` feature to implement [`rand_core::SeedableRng`] and [`rand_core::TryRng`] for all standard generators. This allows interoperability with the `rand` ecosystem (e.g. `rand::Rng`, `rand::SeedableRng`).

```toml
[dependencies]
urng = { version = "0.8.0", features = ["rand"] }
```

> Requires `rand_core = "0.10"`. The `rand` crate itself is only needed as a dev-dependency for tests; consumers only need `urng` with the `rand` feature.

### Optional `simd` Feature

Enable the `simd` feature to build the AVX2/AVX-512 SIMD generators (the `x4`/`x8`/`x16`/`x2`-suffixed structs and the `*Simd` runtime dispatchers, e.g. `Sfc32x4`, `Sfc32x8`, `Sfc32x16`, `Pcg32Simd`). Without it, only the scalar generators are compiled.

```toml
[dependencies]
urng = { version = "0.8.0", features = ["simd"] }
```

> When combined with the `cabi` feature, the SIMD generators' C-ABI exports (e.g. `sfc32x8_new`) are also only available with `simd` enabled.

### Optional `testing` Feature

Enable the `testing` feature for a statistical test harness (chi-squared uniformity test and Monte Carlo π estimation) usable against any `Rng32`/`Rng64` generator, or against `rand`-ecosystem RNGs when combined with the `rand` feature. See [Testing](#testing) below.

```toml
[dependencies]
urng = { version = "0.8.0", features = ["testing"] }
```

When enabled, all scalar generators (`Mt19937`, `Sfmt*`, `Pcg32`, `Sfc32`, `SplitMix32`, `Squares32`, `Xoroshiro64Ss`, `Xorshift32`, `Xorshift128`, `Xorwow`, `Xoshiro128Pp`, `Xoshiro128Ss`) implement:

- [`rand_core::SeedableRng`][SeedableRng] — seed from `[u8; 4]`
- [`rand_core::TryRng`][TryRng] — fallible byte-fill API via [`try_fill_bytes`][try_fill_bytes]

`Philox32x4` (returns `[u32; 4]` per call) implements `TryRng` with a specialized `try_fill_bytes` that packs multiple `u32` outputs per call.

## Supported Generators

Generators are divided into two categories: standard generators and AVX-accelerated SIMD generators.
Standard generators implement either the `Rng32` or `Rng64` trait (or return fixed-size arrays for counter-based variants).
AVX generators expose a bulk-generation API and are listed separately; they require the `simd` feature (see [above](#optional-simd-feature)).

### 32-bit Generators (`urng::rng32`)

| Struct          | Algorithm        | Period / State   |
| --------------- | ---------------- | ---------------- |
| `Mt19937`       | Mersenne Twister | $2^{19937}-1$    |
| `Sfmt607`       | SFMT             | $2^{607}-1$      |
| `Sfmt1279`      | SFMT             | $2^{1279}-1$     |
| `Sfmt2281`      | SFMT             | $2^{2281}-1$     |
| `Sfmt4253`      | SFMT             | $2^{4253}-1$     |
| `Sfmt11213`     | SFMT             | $2^{11213}-1$    |
| `Sfmt19937`     | SFMT             | $2^{19937}-1$    |
| `Sfmt44497`     | SFMT             | $2^{44497}-1$    |
| `Sfmt86243`     | SFMT             | $2^{86243}-1$    |
| `Sfmt132049`    | SFMT             | $2^{132049}-1$   |
| `Sfmt216091`    | SFMT             | $2^{216091}-1$   |
| `Sfc32`         | SFC32            | $2^{127}-1$      |
| `Sfc32x4`       | SFC32 x4         | $2^{127}-1$      |
| `Pcg32`         | PCG-XSH-RR       | $2^{64}$         |
| `Philox32x4`    | Philox 4x32      | -                |
| `SplitMix32`    | SplitMix32       | $2^{32}$         |
| `Xorwow`        | XORWOW           | $2^{192}-2^{32}$ |
| `Xorshift32`    | Xorshift         | $2^{32}-1$       |
| `Xorshift128`   | Xorshift128      | $2^{128}-1$      |
| `Xoshiro128Pp`  | xoshiro128++     | $2^{128}-1$      |
| `Xoshiro128Ss`  | xoshiro128\*\*   | $2^{128}-1$      |
| `Xoroshiro64Ss` | xoroshiro64\*\*  | $2^{64}-1$       |
| `Lcg32`         | LCG              | $m$              |
| `Threefry32x4`  | Threefry 4x32    | -                |
| `Threefry32x2`  | Threefry 2x32    | -                |
| `Squares32`     | Squares          | -                |
| `Jsf32`         | JSF32            | -                |

### 64-bit Generators (`urng::rng64`)

| Struct           | Algorithm           | Period / State   |
| ---------------- | ------------------- | ---------------- |
| `Xoshiro256Pp`   | xoshiro256++        | $2^{256}-1$      |
| `Xoshiro256Ss`   | xoshiro256\*\*      | $2^{256}-1$      |
| `SplitMix64`     | SplitMix64          | $2^{64}$         |
| `Sfc64`          | SFC64               | $2^{256}$ approx |
| `Mt1993764`      | Mersenne Twister 64 | $2^{19937}-1$    |
| `Sfmt1993764`    | SFMT 64             | $2^{19937}-1$    |
| `Philox64`       | Philox 2x64         | -                |
| `Xorshift64`     | Xorshift64          | $2^{64}-1$       |
| `Xoroshiro128Pp` | xoroshiro128++      | $2^{128}-1$      |
| `Xoroshiro128Ss` | xoroshiro128\*\*    | $2^{128}-1$      |
| `TwistedGFSR`    | TGFSR               | $2^{800}$ approx |
| `Cet64`          | CET                 | $2^{64}$         |
| `Cet256`         | CET                 | $2^{256}$        |
| `Lcg64`          | LCG                 | $m$              |
| `Threefish256`   | Threefish-256       | -                |
| `Biski64`        | Biski64             | $2^{64}$         |

### SIMD Generators (AVX)

These generators expose a bulk-generation API and require AVX support at runtime.

#### AVX2 (`avx2`)

| Struct            | Algorithm          | Output  |
| ----------------- | ------------------ | ------- |
| `Sfc32x8`         | SFC32 x8           | 8×`u32` |
| `Jsf32x8`         | JSF32 x8           | 8×`u32` |
| `Xoroshiro64Ssx8` | xoroshiro64\*\* x8 | 8×`u32` |

#### AVX-512 (`avx512f`)

| Struct             | Algorithm           | Output   |
| ------------------ | ------------------- | -------- |
| `Pcg32x8`          | PCG-XSH-RR x8       | 8×`u32`  |
| `Philox32x4x4`     | Philox 4x32 x4      | 16×`u32` |
| `SplitMix32x16`    | SplitMix32 x16      | 16×`u32` |
| `Squares32x8`      | Squares x8          | 8×`u32`  |
| `Xoshiro128Ppx16`  | xoshiro128++ x16    | 16×`u32` |
| `Xoshiro128Ssx16`  | xoshiro128\*\* x16  | 16×`u32` |
| `Jsf32x16`         | JSF32 x16           | 16×`u32` |
| `Sfc32x16`         | SFC32 x16           | 16×`u32` |
| `Xoroshiro64Ssx16` | xoroshiro64\*\* x16 | 16×`u32` |
| `Xoshiro256Ssx2`   | xoshiro256\*\* x2   | 2×`u64`  |
| `Sfc64x8`          | SFC64 x8            | 8×`u64`  |
| `Cet64x8`          | CET64 x8            | 8×`u64`  |
| `Cet256x2`         | CET256 x2           | 2×`u64`  |
| `Biski64x8`        | Biski64 x8          | 8×`u64`  |

## Sampler

> Requires the `sampler` feature.

Weighted random index selection. Two implementations are provided for each bit-width, both implementing the `Sampler32` / `Sampler64` trait (`urng::sampler`).

| Struct    | Module            | Algorithm      | Build | Sample   |
| --------- | ----------------- | -------------- | ----- | -------- |
| `Bst32`   | `urng::sampler32` | Cumulative BST | O(n)  | O(log n) |
| `Alias32` | `urng::sampler32` | Walker's Alias | O(n)  | O(1)     |
| `Bst64`   | `urng::sampler64` | Cumulative BST | O(n)  | O(log n) |
| `Alias64` | `urng::sampler64` | Walker's Alias | O(n)  | O(1)     |

## SeedGen

> Requires the `seedgen` feature.

Hardware-noise-assisted seed generation. Wraps an existing `Rng32`/`Rng64` and mixes in hardware noise (RDSEED/RDRAND on x86/x86_64, timestamp fallback elsewhere) via a Murmur3-style hash.

| Struct      | Module          | Input RNG | Output            |
| ----------- | --------------- | --------- | ----------------- |
| `SeedGen32` | `urng::seedgen` | `Rng32`   | `(u32, u32)` pair |
| `SeedGen64` | `urng::seedgen` | `Rng64`   | `(u64, u64)` pair |

`next_seed_pair()` returns `(raw, processed)` — the raw hardware value and the mixed seed.

## Testing

> Requires the `testing` feature.

A statistical test harness for validating RNG quality, generic over any `Rng32`/`Rng64` implementation.

| Struct                          | Module          | Purpose                              |
| ------------------------------- | --------------- | ------------------------------------ |
| `ChiSq32` / `ChiSq64`           | `urng::testing` | Chi-squared uniformity test          |
| `ChiSqSuite32` / `ChiSqSuite64` | `urng::testing` | Run named chi-squared cases together |
| `McPi32` / `McPi64`             | `urng::testing` | Monte Carlo estimation of π          |
| `McPiSuite32` / `McPiSuite64`   | `urng::testing` | Run named Monte Carlo cases together |
| `Test32` / `Test64`             | `urng::testing` | Blanket trait adding `run_chisq`/`run_mcpi` directly to any `Rng32`/`Rng64` implementor |

```rust
use urng::*;
use urng::testing::{ChiSq32, ChiSqVerdict};

let mut rng = Sfc32::new(0);
let mut chisq = ChiSq32::from_urng(&mut rng);
let result = chisq.run("Sfc32").unwrap();
assert_eq!(result.verdict, ChiSqVerdict::Pass);
```

`Test32`/`Test64` skip the explicit harness construction for the common case — call `run_chisq`/`run_mcpi` straight on the generator:

```rust
use urng::*;
use urng::testing::{ChiSqVerdict, Test32};

let mut rng = Sfc32::new(0);
let result = rng.run_chisq("Sfc32").unwrap();
assert_eq!(result.verdict, ChiSqVerdict::Pass);
```

With the `rand` feature also enabled, `from_rand`/`with_config_from_rand` accept any `rand_core::Rng` implementation directly (via the internal `RandAdapter`), so external generators can be validated with the same harness without manually wrapping them.

## Usage Examples

Most generators expose the same basic workflow: create an instance with `new`, then use `nextu`, `nextf`, `randi`, `randf`, or `choice` depending on the output type you need. SIMD and counter-based generators return fixed-size arrays instead of single values.

Scalar generators (both 32-bit and 64-bit; SIMD variants are not included) also implement `Default`, seeding themselves from a time-based, per-call mix so no explicit seed is required:

```rust
use urng::*;

let mut rng = Sfc32::default();
let _ = rng.nextu();
```

> The deprecated `Lcg32`/`Lcg64` implement `Default` with the fixed parameters `new(8, 13, 5, 24)` instead, since an LCG has no single sensible auto-generated seed.

### Basic Usage

```rust
use urng::*;

fn main() {
    // 1. Initialize with a seed
    let mut rng = Xoshiro256Pp::new(12345);

    // 2. Generate random numbers
    let val_u64 = rng.nextu();
    println!("u64: {}", val_u64);

    let val_f64 = rng.nextf(); // [0.0, 1.0)
    println!("f64: {}", val_f64);

    // 3. Generate within a range
    let val_range = rng.randi(1, 100);
    println!("Integer (1-100): {}", val_range);

    // 4. Seeding with SplitMix64 (common pattern)
    // If you need to seed a large state generator from a single u64
    let mut sm = SplitMix64::new(9999);
    let seed_val = sm.nextu();
    let mut rng2 = Xoshiro256Pp::new(seed_val);
}
```

## C ABI

This crate exports a C-compatible ABI generic interface. Each generator has corresponding:

- `_new`
- `_free`
- `_next_uXXs` (bulk generation)
- `_next_fXXs` (bulk generation)
- `_rand_iXXs` (bulk generation)
- `_rand_fXXs` (bulk generation)

Example for `Mt19937`:

```c
void* mt19937_new(uint32_t seed, size_t warm);
void mt19937_next_u32s(void* ptr, uint32_t* out, size_t count);
void mt19937_rand_f32s(void* ptr, float* out, size_t count, float min, float max);
void mt19937_free(void* ptr);
```

[SeedableRng]: https://docs.rs/rand_core/latest/rand_core/trait.SeedableRng.html
[TryRng]: https://docs.rs/rand_core/latest/rand_core/trait.TryRng.html
[try_fill_bytes]: https://docs.rs/rand_core/latest/rand_core/trait.TryRng.html#tymethod.try_fill_bytes
