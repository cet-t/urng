//! Statistical test harness for PRNG quality evaluation.
//!
//! Available in **debug builds only** (`cargo build` / `cargo test`).
//! Not compiled into release builds or the distributed `cdylib`.
//!
//! # Modules
//!
//! - [`stream`] — [`ByteStream32`](stream::ByteStream32) /
//!   [`ByteStream64`](stream::ByteStream64): adapts any RNG into [`std::io::Read`].
//! - [`pracrand`] — pipes RNG output into a `RNG_test` subprocess.
//!   Requires `RNG_test` on `PATH` or `PRACRAND_PATH` set.
//! - [`testu01`] — calls TestU01 batteries via FFI.
//!   Compiled only when `TESTU01_LIB_DIR` is set at build time (`has_testu01` cfg).
//!
//! # Migrating to a feature flag
//!
//! Replace the gate in `lib.rs` with:
//! ```toml
//! # Cargo.toml
//! [features]
//! testing = []
//! ```
//! ```rust,ignore
//! // lib.rs
//! #[cfg(any(debug_assertions, feature = "testing"))]
//! pub mod testing;
//! ```

pub mod stream;
pub mod pracrand;

#[cfg(has_testu01)]
pub mod testu01;
