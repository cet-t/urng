//! Counter-based 32-bit random number generators.

pub(crate) mod philox;
pub(crate) mod squares;
pub(crate) mod threefry;

pub use philox::Philox32;
#[cfg(feature = "simd")]
pub use philox::Philox32x4x4;
pub use squares::Squares32;
#[cfg(feature = "simd")]
pub use squares::Squares32x8;
pub use threefry::{Threefry32x2, Threefry32x4};

#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use philox::{
    PHILOX32x4x4_CHUNK_RATIO, PHILOX32x4x4_PAR_CHUNK, PHILOX32x4x4_SHIFT, PHILOX32x16,
    PHILOX32x16_SHIFT,
};
#[cfg(all(feature = "cabi", feature = "simd"))]
pub(crate) use squares::SQUARES32x8;

crate::impl_default_from_seed32!(Squares32, Philox32, Threefry32x2, Threefry32x4,);
