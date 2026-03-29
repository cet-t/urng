use crate::rng32::{
    Mt19937, Sfmt521, Sfmt1279, Sfmt2203, Sfmt4253, Sfmt11213, Sfmt19937, Sfmt44497, Sfmt86243,
    Sfmt132049, Sfmt216091,
};
use std::slice::from_raw_parts_mut;

/// Creates a new `Mt19937` instance.
/// The caller is responsible for freeing the memory using `mt19937_free`.
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_new(seed: u32) -> *mut Mt19937 {
    Box::into_raw(Box::new(Mt19937::new(seed)))
}

/// Frees the memory of a `Mt19937` instance.
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_free(ptr: *mut Mt19937) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_next_u32s(ptr: *mut Mt19937, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        rng.fill_next_u32s(buffer);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_next_f32s(ptr: *mut Mt19937, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_rand_i32s(
    ptr: *mut Mt19937,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn mt19937_rand_f32s(
    ptr: *mut Mt19937,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

/// Creates a new `Sfmt19937` instance.
/// The caller is responsible for freeing the memory using `sfmt19937_free`.
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_new(seed: u64) -> *mut Sfmt19937 {
    Box::into_raw(Box::new(Sfmt19937::new(seed)))
}

/// Frees the memory of a `Sfmt19937` instance.
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_free(ptr: *mut Sfmt19937) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

/// Fills the output buffer with the next random `u32` values.
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_next_u32s(ptr: *mut Sfmt19937, out: *mut u32, count: usize) {
    if count == 0 {
        return;
    }
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        rng.fill_next_u32s(buffer);
    }
}

/// Fills the output buffer with the next random `f32` values in the range [0, 1).
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_next_f32s(ptr: *mut Sfmt19937, out: *mut f32, count: usize) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.nextf();
        }
    }
}

/// Fills the output buffer with random `i32` values in the range [min, max].
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_rand_i32s(
    ptr: *mut Sfmt19937,
    out: *mut i32,
    count: usize,
    min: i32,
    max: i32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randi(min, max);
        }
    }
}

/// Fills the output buffer with random `f32` values in the range [min, max).
#[unsafe(no_mangle)]
pub extern "C" fn sfmt19937_rand_f32s(
    ptr: *mut Sfmt19937,
    out: *mut f32,
    count: usize,
    min: f32,
    max: f32,
) {
    unsafe {
        let rng = &mut *ptr;
        let buffer = from_raw_parts_mut(out, count);
        for x in buffer {
            *x = rng.randf(min, max);
        }
    }
}

macro_rules! impl_sfmt_cabi {
    (
        $ty:ty,
        $new_fn:ident,
        $free_fn:ident,
        $next_u32s_fn:ident,
        $next_f32s_fn:ident,
        $rand_i32s_fn:ident,
        $rand_f32s_fn:ident
    ) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $new_fn(seed: u64) -> *mut $ty {
            Box::into_raw(Box::new(<$ty>::new(seed)))
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn $free_fn(ptr: *mut $ty) {
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn $next_u32s_fn(ptr: *mut $ty, out: *mut u32, count: usize) {
            if count == 0 {
                return;
            }
            unsafe {
                let rng = &mut *ptr;
                let buffer = from_raw_parts_mut(out, count);
                rng.fill_next_u32s(buffer);
            }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn $next_f32s_fn(ptr: *mut $ty, out: *mut f32, count: usize) {
            unsafe {
                let rng = &mut *ptr;
                let buffer = from_raw_parts_mut(out, count);
                for x in buffer {
                    *x = rng.nextf();
                }
            }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn $rand_i32s_fn(
            ptr: *mut $ty,
            out: *mut i32,
            count: usize,
            min: i32,
            max: i32,
        ) {
            unsafe {
                let rng = &mut *ptr;
                let buffer = from_raw_parts_mut(out, count);
                for x in buffer {
                    *x = rng.randi(min, max);
                }
            }
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn $rand_f32s_fn(
            ptr: *mut $ty,
            out: *mut f32,
            count: usize,
            min: f32,
            max: f32,
        ) {
            unsafe {
                let rng = &mut *ptr;
                let buffer = from_raw_parts_mut(out, count);
                for x in buffer {
                    *x = rng.randf(min, max);
                }
            }
        }
    };
}

impl_sfmt_cabi!(
    Sfmt521,
    sfmt521_new,
    sfmt521_free,
    sfmt521_next_u32s,
    sfmt521_next_f32s,
    sfmt521_rand_i32s,
    sfmt521_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt1279,
    sfmt1279_new,
    sfmt1279_free,
    sfmt1279_next_u32s,
    sfmt1279_next_f32s,
    sfmt1279_rand_i32s,
    sfmt1279_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt2203,
    sfmt2203_new,
    sfmt2203_free,
    sfmt2203_next_u32s,
    sfmt2203_next_f32s,
    sfmt2203_rand_i32s,
    sfmt2203_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt4253,
    sfmt4253_new,
    sfmt4253_free,
    sfmt4253_next_u32s,
    sfmt4253_next_f32s,
    sfmt4253_rand_i32s,
    sfmt4253_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt11213,
    sfmt11213_new,
    sfmt11213_free,
    sfmt11213_next_u32s,
    sfmt11213_next_f32s,
    sfmt11213_rand_i32s,
    sfmt11213_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt44497,
    sfmt44497_new,
    sfmt44497_free,
    sfmt44497_next_u32s,
    sfmt44497_next_f32s,
    sfmt44497_rand_i32s,
    sfmt44497_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt86243,
    sfmt86243_new,
    sfmt86243_free,
    sfmt86243_next_u32s,
    sfmt86243_next_f32s,
    sfmt86243_rand_i32s,
    sfmt86243_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt132049,
    sfmt132049_new,
    sfmt132049_free,
    sfmt132049_next_u32s,
    sfmt132049_next_f32s,
    sfmt132049_rand_i32s,
    sfmt132049_rand_f32s
);

impl_sfmt_cabi!(
    Sfmt216091,
    sfmt216091_new,
    sfmt216091_free,
    sfmt216091_next_u32s,
    sfmt216091_next_f32s,
    sfmt216091_rand_i32s,
    sfmt216091_rand_f32s
);
