//! C-compatible ABI wrappers for the stateful PRNGs.

#![allow(ambiguous_glob_reexports)]

pub mod b32;
pub mod b64;

pub use b32::*;
pub use b64::*;
