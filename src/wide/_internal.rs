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

pub(crate) use wide_rotate_left;
pub(crate) use wide_rotate_right;
