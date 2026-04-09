const INV_2_POW_32: f64 = 1.0 / 4294967296.0;
const INV_2_POW_53: f64 = 1.0 / 9007199254740992.0;

#[inline(always)]
pub const fn unit_f64_from_u32(x: u32) -> f64 {
    x as f64 * INV_2_POW_32
}

#[inline(always)]
pub const fn unit_f64_from_u64(x: u64) -> f64 {
    ((x >> 11) as f64) * INV_2_POW_53
}
