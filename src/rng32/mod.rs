pub mod lcg;
pub mod mersenne;
pub mod pcg;
pub mod philox;
pub mod splitmix;
pub mod squares;
pub mod threefry;
pub mod xorshift;

#[allow(deprecated)]
pub use lcg::Lcg32;
pub use mersenne::{Mt19937, Sfmt19937};
pub(crate) use pcg::{PCG32_MULT, PCG32X8_LANE, PCG32X8_PAR_CHUNK, PCG32X8_PAR_CHUNK_BLOCKS};
pub use pcg::{Pcg32, Pcg32Simd, Pcg32x8};
pub(crate) use philox::{
    PHILOX32x4x4_CHUNK_RATIO, PHILOX32x4x4_PAR_CHUNK, PHILOX32x4x4_SHIFT, PHILOX32x16,
    PHILOX32x16_SHIFT,
};
pub use philox::{Philox32, Philox32x4, Philox32x4x4};
pub(crate) use splitmix::{SPLITMIX32_GAMMA, SPLITMIX32x16, SPLITMIX32x16_PAR_CHUNK};
pub use splitmix::{SplitMix32, SplitMix32Simd, SplitMix32x16};
pub use squares::{SQUARES32x8, Squares32, Squares32Simd, Squares32x8};
pub use threefry::{Threefry32x2, Threefry32x4};
pub use xorshift::{Xorshift32, Xorwow};
