mod _internal;
mod jsf;
mod pcg;
mod sfc;
mod splitmix;
mod squares;
mod threefry;
mod xoroshiro;
mod xorshift;
mod xoshiro;

pub(super) use _internal::*;

pub use jsf::*;
pub use pcg::*;
pub use sfc::*;
pub use splitmix::*;
pub use squares::*;
pub use threefry::*;
pub use xoroshiro::*;
pub use xorshift::*;
pub use xoshiro::*;

crate::impl_default_from_seed32!(
    Jsf32x4,
    Jsf32x8,
    Jsf32x16,
    Sfc32x4,
    Sfc32x8,
    Sfc32x16,
    SplitMix32x4,
    SplitMix32x8,
    SplitMix32x16,
    Threefry32x2x4,
    Threefry32x2x8,
    Threefry32x2x16,
    Threefry32x4x4,
    Threefry32x4x8,
    Threefry32x4x16,
    Xoroshiro64Ssx4,
    Xoroshiro64Ssx8,
    Xoroshiro64Ssx16,
    Xorshift32x4,
    Xorshift32x8,
    Xorshift32x16,
    Xorshift128x4,
    Xorshift128x8,
    Xorshift128x16,
    Xorwowx4,
    Xorwowx8,
    Xorwowx16,
    Xoshiro128Ppx4,
    Xoshiro128Ppx8,
    Xoshiro128Ppx16,
    Xoshiro128Ssx4,
    Xoshiro128Ssx8,
    Xoshiro128Ssx16,
);

crate::impl_default_from_seed64!(
    Pcg32x4,
    Pcg32x8,
    Pcg32x16,
    Squares32x4,
    Squares32x8,
    Squares32x16,
);
