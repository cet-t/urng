mod sfc;
mod splitmix;

pub use sfc::Sfc32x8;
pub use splitmix::{SplitMix32x4, SplitMix32x8};

pub mod prelude {
    pub use super::{Sfc32x8, SplitMix32x4, SplitMix32x8};
}
