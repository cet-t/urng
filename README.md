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
urng = "0.4.5"
```

## Supported Generators

Most generators implement either the `Rng32` or `Rng64` trait, providing a unified interface.
SIMD generators have their own bulk-generation API and are listed separately.

### 32-bit Generators (`urng::rng32`)

Implement `Rng32`, output `u32` natively.

| Struct         | Algorithm        | Period / State   | Description                                     |
| -------------- | ---------------- | ---------------- | ----------------------------------------------- |
| `Mt19937`      | Mersenne Twister | $2^{19937}-1$    | Standard reliable generator.                    |
| `Sfmt607`      | SFMT             | $2^{607}-1$      | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt1279`     | SFMT             | $2^{1279}-1$     | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt2281`     | SFMT             | $2^{2281}-1$     | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt4253`     | SFMT             | $2^{4253}-1$     | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt11213`    | SFMT             | $2^{11213}-1$    | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt19937`    | SFMT             | $2^{19937}-1$    | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt44497`    | SFMT             | $2^{44497}-1$    | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt86243`    | SFMT             | $2^{86243}-1$    | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt132049`   | SFMT             | $2^{132049}-1$   | SIMD-oriented Fast Mersenne Twister.            |
| `Sfmt216091`   | SFMT             | $2^{216091}-1$   | SIMD-oriented Fast Mersenne Twister.            |
| `Pcg32`        | PCG-XSH-RR       | $2^{64}$         | Fast, statistically good, small state.          |
| `Philox32x4`   | Philox4x32-10    | -                | Counter-based, suitable for parallel use.       |
| `SplitMix32`   | SplitMix32       | $2^{32}$         | Fast, used for initializing other states.       |
| `Xorwow`       | XORWOW           | $2^{192}-2^{32}$ | Used in NVIDIA cuRAND.                          |
| `Xorshift32`   | Xorshift         | $2^{32}-1$       | Very simple and fast.                           |
| `Xorshift128`  | Xorshift         | $2^{32}-1$       | Very simple and fast.                           |
| `Xoshiro128Pp` | xoshiro128++     | $2^{128}-1$      | Fast high-quality generator with 128-bit state. |
| `Xoshiro128Ss` | xoshiro128\*\*   | $2^{128}-1$      | Fast high-quality generator with 128-bit state. |
| `Lcg32`        | LCG              | $m$              | Linear Congruential Generator.                  |
| `Threefry32x4` | Threefry 4x32    | -                | Counter-based (Random123 family).               |
| `Threefry32x2` | Threefry 2x32    | -                | Counter-based (Random123 family).               |
| `Squares32`    | Squares          | -                | Counter-based (Widynski).                       |
| `Jsf32`        | JSF32            | -                | Jenskin Small Fast.                             |

### 64-bit Generators (`urng::rng64`)

Implement `Rng64`, output `u64` natively.

| Struct            | Algorithm           | Period / State   | Description                                   |
| ----------------- | ------------------- | ---------------- | --------------------------------------------- |
| `Xoshiro256Pp`    | xoshiro256++        | $2^{256}-1$      | **Recommended** all-purpose generator.        |
| `Xoshiro256Ss`    | xoshiro256\*\*      | $2^{256}-1$      | **Recommended** all-purpose generator.        |
| `SplitMix64`      | SplitMix64          | $2^{64}$         | Fast, used for initializing other states.     |
| `Sfc64`           | SFC64               | $2^{256}$ approx | Small Fast Chaotic PRNG.                      |
| `Mt1993764`       | Mersenne Twister 64 | $2^{19937}-1$    | 64-bit variant of MT.                         |
| `Sfmt1993764`     | SFMT 64             | $2^{19937}-1$    | SIMD-oriented Fast Mersenne Twister.          |
| `Philox64`        | Philox 2x64         | -                | Counter-based.                                |
| `Xorshift64`      | Xorshift            | $2^{64}-1$       | Simple and fast.                              |
| `Xoroshiro128Pp`¹ | xoroshiro128++      | $2^{128}-1$      | Fast generator with 128-bit state.            |
| `Xoroshiro128Ss`¹ | xoroshiro128\*\*    | $2^{128}-1$      | Fast generator with 128-bit state.            |
| `TwistedGFSR`     | TGFSR               | $2^{800}$ approx | Generalized Feedback Shift Register.          |
| `Cet64`           | CET                 | $2^{64}$         | Custom experimental generator.                |
| `Cet256`          | CET                 | $2^{256}$        | Custom experimental generator.                |
| `Lcg64`           | LCG                 | $m$              | Linear Congruential Generator.                |
| `Threefish256`    | Threefish-256       | -                | Counter-based, 256-bit block cipher PRNG.     |
| `Biski64`         | Biski64             | $2^{64}$         | Extremely fast pseudo-random number generator |

> ¹ In the `urng::rng64::xoroshiro` submodule (not re-exported at `urng::rng64`).

### SIMD Generators

These generators do not implement `Rng32`/`Rng64` and instead expose a bulk-generation API.

#### AVX-512 (`avx512f`)

| Struct            | Algorithm          | Output   | Description                              |
| ----------------- | ------------------ | -------- | ---------------------------------------- |
| `Pcg32x8`         | PCG-XSH-RR x8      | 8×`u32`  | 8 independent PCG32 streams in parallel. |
| `Philox32x4x4`    | Philox 4x32 x4     | 16×`u32` | Counter-based, 4 Philox4x32 streams.     |
| `SplitMix32x16`   | SplitMix32 x16     | 16×`u32` | 16 independent SplitMix32 streams.       |
| `Squares32x8`     | Squares x8         | 8×`u32`  | 8 counters processed in parallel.        |
| `Xoshiro128Ppx16` | xoshiro128++ x16   | 16×`u32` | 16 independent xoshiro128++ streams.     |
| `Xoshiro128Ssx16` | xoshiro128\*\* x16 | 16×`u32` | 16 independent xoshiro128\*\* streams.   |
| `Jsf32x16`        | JSF32 x16          | 16×`u32` | 16 independent JSF32 streams.            |
| `Xoshiro256Ssx2`  | xoshiro256\*\* x2  | 2×`u64`  | 2 independent xoshiro256\*\* streams.    |
| `Sfc64x8`         | SFC64 x8           | 8×`u64`  | 8 independent SFC64 streams.             |
| `Cet64x8`         | CET64 x8           | 8×`u64`  | 8 independent CET64 streams.             |
| `Cet256x2`        | CET256 x2          | 2×`u64`  | 8 independent CET256 streams.            |

## Sampler

> Requires the `sampler` feature.

Weighted random index selection. Two implementations are provided for each bit-width, both implementing the `Sampler32` / `Sampler64` trait (`urng::sampler`).

| Struct    | Module            | Algorithm      | Build | Sample   | Description                                    |
| --------- | ----------------- | -------------- | ----- | -------- | ---------------------------------------------- |
| `Bst32`   | `urng::sampler32` | Cumulative BST | O(n)  | O(log n) | Binary-search over cumulative weights (`f32`). |
| `Alias32` | `urng::sampler32` | Walker's Alias | O(n)  | O(1)     | Preferred for repeated sampling (`f32`).       |
| `Bst64`   | `urng::sampler64` | Cumulative BST | O(n)  | O(log n) | Binary-search over cumulative weights (`f64`). |
| `Alias64` | `urng::sampler64` | Walker's Alias | O(n)  | O(1)     | Preferred for repeated sampling (`f64`).       |

## SeedGen

> Requires the `seedgen` feature.

Hardware-noise-assisted seed generation. Wraps an existing `Rng32`/`Rng64` and mixes in hardware noise (RDSEED/RDRAND on x86/x86_64, timestamp fallback elsewhere) via a Murmur3-style hash.

| Struct      | Module          | Input RNG | Output            | Description                             |
| ----------- | --------------- | --------- | ----------------- | --------------------------------------- |
| `SeedGen32` | `urng::seedgen` | `Rng32`   | `(u32, u32)` pair | 32-bit hardware-noise-assisted seeding. |
| `SeedGen64` | `urng::seedgen` | `Rng64`   | `(u64, u64)` pair | 64-bit hardware-noise-assisted seeding. |

`next_seed_pair()` returns `(raw, processed)` — the raw hardware value and the mixed seed.

## Usage Examples

Most generators expose the same basic workflow: create an instance with `new`, then use `nextu`, `nextf`, `randi`, `randf`, or `choice` depending on the output type you need. SIMD and counter-based generators return fixed-size arrays instead of single values.

### Basic Usage

```rust
use urng::prelude::*;

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

Example for `Mt19937`:

```c
void* mt19937_new(uint32_t seed, size_t warm);
void mt19937_next_u32s(void* ptr, uint32_t* out, size_t count);
void mt19937_free(void* ptr);
```
