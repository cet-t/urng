# Universal RNG

A collection of pseudo-random number generators (PRNGs) implemented in Rust.

This crate provides efficient implementations of various random number generation algorithms, supporting both Rust and C (via FFI).

## Supported Generators

- **Mersenne Twister**:
  - `mt19937::Mt19937` (32-bit)
  - `mt1993764::Mt1993764` (64-bit)
- **PCG**:
  - `pcg32::Pcg32` (Permuted Congruential Generator, PCG-XSH-RR)
- **Philox** (Counter-based RNGs):
  - `philox32::Philox32` (4x32)
  - `philox64::Philox64` (2x64)
- **Twisted GFSR**:
  - `twisted_gfsr::TwistedGFSR`
- **Xorshift**:
  - `xorshift32::Xorshift32`
  - `xorshift64::Xorshift64`
  - `xorshift128::Xorshift128`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
urng = "0.1.0"
```

## Usage Examples

### Basic Usage (Mersenne Twister)

```rust
use urng::mt19937::Mt19937;

fn main() {
    // Initialize with a seed
    let mut rng = Mt19937::new(12345);

    // Generate a random u32
    let val = rng.nextu();
    println!("Random u32: {}", val);

    // Generate a random float in [0, 1)
    let f_val = rng.nextf();
    println!("Random f32: {}", f_val);

    // Generate a random integer in a range [min, max]
    let range_val = rng.randi(1, 100);
    println!("Random integer (1-100): {}", range_val);
}
```

### Using Philox (Counter-based)

Philox generators produce blocks of random numbers and are suitable for parallel applications.

```rust
use urng::philox32::Philox32;

fn main() {
    let mut rng = Philox32::new([123, 456]);

    // Generates 4 x u32 values at once
    let values = rng.nextu();
    println!("Random values: {:?}", values);
}
```

## C API

This crate exports C-compatible functions for all generators, allowing them to be used as a shared library.

Each generator exposes functions for:

- Creation (`_new`)
- Destruction (`_free`)
- Bulk generation (`_next_uXXs`, `_next_fXXs`)
- Range generation (`_rand_iXXs`, `_rand_fXXs`)

Example signature for MT19937:

```c
void* mt19937_new(uint32_t seed);
void mt19937_next_u32s(void* ptr, uint32_t* out, size_t count);
void mt19937_free(void* ptr);
```
