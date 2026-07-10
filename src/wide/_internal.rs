#[macro_export]
macro_rules! i2f_bits {
    (32 bits) => {
        0x3F800000
    };
    (32 bias) => {
        9
    };
    (64 bits) => {
        0x3FF0000000000000
    };
    (64 bias) => {
        11
    };
}

macro_rules! impl_methods {
    ($size:expr, $bits:tt) => {
        ::paste::paste! {
            #[inline(always)]
            pub fn nextf(&mut self) -> [[<f $bits>]; $size] {
                self.nextu()
                    .map(|x| [<f $bits>]::from_bits((x >> $crate::i2f_bits!($bits bias)) | $crate::i2f_bits!($bits bits)) - 1.0)
            }

            #[inline(always)]
            pub fn randi(&mut self, min: [<i $bits>], max: [<i $bits>]) -> [[<i $bits>]; $size] {
                let range = (max as i128 - min as i128 + 1) as u128;
                self.nextu().map(|x| (((x as u128) * range >> $bits) as [<i $bits>]) + min)
            }

            #[inline(always)]
            pub fn randf(&mut self, min: [<f $bits>], max: [<f $bits>]) -> [[<f $bits>]; $size] {
                self.nextu().map(|x| {
                    let base = [<f $bits>]::from_bits(
                        (x >> $crate::i2f_bits!($bits bias)) | $crate::i2f_bits!($bits bits),
                    ) - 1.0;
                    base * (max - min) + min
                })
            }
        }
    };
}

macro_rules! wide_rotate_left {
    (32 $x:expr, $shift:expr) => {
        ($x << $shift) | ($x >> (32 - $shift))
    };
    (64 $x:expr, $shift:expr) => {
        ($x << $shift) | ($x >> (64 - $shift))
    };
}

macro_rules! wide_rotate_right {
    (32 $x:expr, $shift:expr) => {
        ($x >> $shift) | ($x << (32 - $shift))
    };
    (64 $x:expr, $shift:expr) => {
        ($x >> $shift) | ($x << (64 - $shift))
    };
}

pub(crate) use impl_methods;
pub(crate) use wide_rotate_left;
pub(crate) use wide_rotate_right;
