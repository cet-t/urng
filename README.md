# Universal RNG

A collection of efficient pseudo-random number generators (PRNGs) implemented in pure Rust.
This crate provides a wide variety of algorithms, ranging from standard Mersenne Twister to modern high-performance generators like Xoshiro and Philox.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
urng = "0.2.0"
```

## Supported Generators

All generators implement either the `Rng32` or `Rng64` trait, providing a unified interface.

### 32-bit Generators (`urng::rng32`)

Output `u32` natively.

| Struct        | Algorithm        | Period / State   | Description                               |
| ------------- | ---------------- | ---------------- | ----------------------------------------- |
| `Mt19937`     | Mersenne Twister | $2^{19937}-1$    | Standard reliable generator.              |
| `Pcg32`       | PCG-XSH-RR       | $2^{64}$         | Fast, statistically good, small state.    |
| `Philox32`    | Philox 4x32      | -                | Counter-based, suitable for parallel use. |
| `Xorwow`      | XORWOW           | $2^{192}-2^{32}$ | Used in NVIDIA cuRAND.                    |
| `Xorshift32`  | Xorshift         | $2^{32}-1$       | Very simple and fast.                     |
| `TwistedGFSR` | TGFSR            | $2^{800}$ approx | Generalized Feedback Shift Register.      |
| `Lcg32`       | LCG              | $m$              | Linear Congruential Generator.            |

### 64-bit Generators (`urng::rng64`)

Output `u64` natively.

| Struct         | Algorithm           | Period / State   | Description                               |
| -------------- | ------------------- | ---------------- | ----------------------------------------- |
| `Xoshiro256Pp` | xoshiro256++        | $2^{256}-1$      | **Recommended** all-purpose generator.    |
| `Xoshiro256Ss` | xoshiro256\*\*      | $2^{256}-1$      | **Recommended** all-purpose generator.    |
| `SplitMix64`   | SplitMix64          | $2^{64}$         | Fast, used for initializing other states. |
| `Sfc64`        | SFC64               | $2^{256}$ approx | Small Fast Chaotic PRNG.                  |
| `Mt1993764`    | Mersenne Twister 64 | $2^{19937}-1$    | 64-bit variant of MT.                     |
| `Philox64`     | Philox 2x64         | -                | Counter-based.                            |
| `Xorshift64`   | Xorshift            | $2^{64}-1$       | Simple and fast.                          |
| `Cet64`        | CET                 | -                | Custom experimental generator.            |
| `Lcg64`        | LCG                 | $m$              | Linear Congruential Generator.            |

### Other (`urng::rng128`)

- `Xorshift128`: 128-bit state Xorshift, implements `Rng32` (outputs `u32`).

## Usage Examples

### Basic Usage

```rust
use urng::rng64::{Xoshiro256Pp, SplitMix64};
use urng::rng::Rng64; // Import trait for common methods

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

### Weighted Choice (BST)

The crate includes a binary-search-tree based weighted selector for efficient sampling.

```rust
use urng::bst::choice;
use urng::rng64::Xoshiro256Pp;

fn main() {
    let mut rng = Xoshiro256Pp::new(42);
    let items = vec!["Common", "Rare", "Epic"];
    let weights = vec![100.0, 10.0, 1.0];

    // Select an item based on weights
    if let Some(item) = choice(&mut rng, weights, &items) {
        println!("You got: {}", item);
    }
}
```

### Using Macros

For convenience, you can use the provided macros (must import `urng::*`).
These macros automatically initialize the generator with a system-seeded state.

```rust
use urng::{next, rand}; // Import macros

fn main() {
    // next! generates a single random number
    // Format: [algorithm][bits][u/f]
    let u = next!(mt64u); // Returns u64 using Mersenne Twister 64
    let f = next!(xor32f); // Returns f32 using Xorshift 32

    // rand! generates a number within a range
    // Format: [algorithm][bits][i/f]; [min], [max]
    let r = rand!(xor32i; 1, 10); // Returns i32 in range [1, 10]
    let f2 = rand!(mt64f; 0.0, 1.0); // Returns f64 in range [0.0, 1.0)
}
```

## C API

This crate exports a C-compatible FFI generic interface. Each generator has corresponding:

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
