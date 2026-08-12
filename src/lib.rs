#![doc = include_str!("../README.md")]
#![allow(ambiguous_glob_reexports)]

mod rng;
pub use crate::rng::*;

#[cfg(feature = "simd")]
mod rngv;
#[cfg(feature = "simd")]
pub use crate::rngv::*;

mod prng;
pub use crate::prng::*;

mod cbrng;
pub use crate::cbrng::*;

mod crng;
pub use crate::crng::*;

#[cfg(feature = "rand")]
mod rand;

#[cfg(feature = "cabi")]
pub mod cabi;
#[cfg(feature = "cabi")]
pub use crate::cabi::{cbrng::*, prng::*};

mod shuffle;
pub use crate::shuffle::*;

mod choice;
pub use crate::choice::*;

pub(crate) mod _internal;
pub(crate) use crate::_internal::*;

#[cfg(feature = "wide")]
pub mod wide;

#[cfg(feature = "sampler")]
mod sampler;
#[cfg(feature = "sampler")]
pub use crate::sampler::*;

#[cfg(feature = "seedgen")]
mod seedgen;
#[cfg(feature = "seedgen")]
pub use crate::seedgen::*;

#[macro_use]
mod macros;
