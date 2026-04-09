use crate::rng::{Rng32, Rng64};
use std::cell::RefCell;
use std::ffi::{CString, c_char, c_double, c_int, c_ulong, c_void};
use std::rc::Rc;
use std::sync::Mutex;

// ── FFI ───────────────────────────────────────────────────────────────────────

/// Matches `unif01_Gen` from `unif01.h` exactly (field order matters for C ABI).
#[repr(C)]
struct Unif01Gen {
    name:     *mut c_char,
    get_u01:  Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_double>,
    get_bits: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_ulong>,
    write:    Option<unsafe extern "C" fn(*mut c_void)>,
    param:    *mut c_void,
    state:    *mut c_void,
}

// SAFETY: only used while holding LOCK; never accessed concurrently.
unsafe impl Send for Unif01Gen {}

unsafe extern "C" {
    fn bbattery_SmallCrush(ugen: *mut Unif01Gen);
    fn bbattery_Crush(ugen: *mut Unif01Gen);
    fn bbattery_BigCrush(ugen: *mut Unif01Gen);

    // TestU01 sets these globals after each battery run (1-based index).
    static bbattery_NTests: c_int;
    static bbattery_pVal: c_double; // first element; walked via pointer arithmetic
}

// ── thread-local callbacks ────────────────────────────────────────────────────
//
// Rng32/Rng64 have a generic method `choice<T>` which makes them not dyn
// compatible. We store plain FnMut closures instead, sharing the RNG between
// get_u01 and get_bits via Rc<RefCell>.

thread_local! {
    static U01_32:  RefCell<Option<Box<dyn FnMut() -> f64>>>    = const { RefCell::new(None) };
    static BITS_32: RefCell<Option<Box<dyn FnMut() -> c_ulong>>> = const { RefCell::new(None) };
    static U01_64:  RefCell<Option<Box<dyn FnMut() -> f64>>>    = const { RefCell::new(None) };
    static BITS_64: RefCell<Option<Box<dyn FnMut() -> c_ulong>>> = const { RefCell::new(None) };
}

unsafe extern "C" fn get_u01_32(_p: *mut c_void, _s: *mut c_void) -> c_double {
    U01_32.with(|c| c.borrow_mut().as_mut().unwrap()())
}
unsafe extern "C" fn get_bits_32(_p: *mut c_void, _s: *mut c_void) -> c_ulong {
    BITS_32.with(|c| c.borrow_mut().as_mut().unwrap()())
}
unsafe extern "C" fn get_u01_64(_p: *mut c_void, _s: *mut c_void) -> c_double {
    U01_64.with(|c| c.borrow_mut().as_mut().unwrap()())
}
unsafe extern "C" fn get_bits_64(_p: *mut c_void, _s: *mut c_void) -> c_ulong {
    BITS_64.with(|c| c.borrow_mut().as_mut().unwrap()())
}
unsafe extern "C" fn write_noop(_s: *mut c_void) {}

// ── public types ──────────────────────────────────────────────────────────────

/// TestU01 battery to run.
#[derive(Clone, Copy, Debug)]
pub enum Battery {
    SmallCrush,
    Crush,
    BigCrush,
}

impl Battery {
    fn name(self) -> &'static str {
        match self {
            Battery::SmallCrush => "SmallCrush",
            Battery::Crush => "Crush",
            Battery::BigCrush => "BigCrush",
        }
    }
}

/// Result of a TestU01 battery run.
pub struct TestResult {
    pub battery: &'static str,
    /// Total number of tests in the battery.
    pub n_tests: usize,
    /// Number of tests that failed (p < α or p > 1−α).
    pub n_failed: usize,
    /// p-values indexed from 1 (TestU01 convention); index 0 is unused (0.0).
    pub p_values: Vec<f64>,
}

impl TestResult {
    pub fn passed(&self) -> bool {
        self.n_failed == 0
    }
}

// ── internal ──────────────────────────────────────────────────────────────────

/// Significance level. TestU01 convention: suspect if p < α or p > 1−α.
const ALPHA: f64 = 0.001;

/// Serialize all battery runs — `bbattery_NTests`/`bbattery_pVal` are globals.
static LOCK: Mutex<()> = Mutex::new(());

fn collect(battery: Battery) -> TestResult {
    let n = unsafe { bbattery_NTests } as usize;
    let base = &raw const bbattery_pVal as *const f64;
    // index 0 unused; collect indices 0..=n so callers can use 1-based indexing
    let p_values: Vec<f64> = (0..=n).map(|i| unsafe { *base.add(i) }).collect();
    let n_failed = p_values[1..]
        .iter()
        .filter(|&&p| p >= 0.0 && (p < ALPHA || p > 1.0 - ALPHA))
        .count();
    TestResult { battery: battery.name(), n_tests: n, n_failed, p_values }
}

unsafe fn run_battery(ugen: *mut Unif01Gen, battery: Battery) {
    unsafe {
        match battery {
            Battery::SmallCrush => bbattery_SmallCrush(ugen),
            Battery::Crush      => bbattery_Crush(ugen),
            Battery::BigCrush   => bbattery_BigCrush(ugen),
        }
    }
}

fn clear_callbacks_32() {
    U01_32.with(|c| *c.borrow_mut() = None);
    BITS_32.with(|c| *c.borrow_mut() = None);
}
fn clear_callbacks_64() {
    U01_64.with(|c| *c.borrow_mut() = None);
    BITS_64.with(|c| *c.borrow_mut() = None);
}

// ── public API ────────────────────────────────────────────────────────────────

/// Run a TestU01 battery on a [`Rng32`] generator.
///
/// TestU01 prints per-test results directly to stdout.
///
/// # Example
/// ```no_run
/// use urng::prelude::*;
/// use urng::testing::testu01::{self, Battery};
///
/// let rng = Xoshiro256Pp::new(42);
/// let result = testu01::run32(rng, Battery::SmallCrush);
/// assert!(result.passed(), "{} failed {}/{}", result.battery, result.n_failed, result.n_tests);
/// ```
pub fn run32<R: Rng32 + 'static>(rng: R, battery: Battery) -> TestResult {
    let _guard = LOCK.lock().expect("testu01 mutex poisoned");

    let rng = Rc::new(RefCell::new(rng));
    let rng2 = Rc::clone(&rng);
    U01_32.with(|c| {
        *c.borrow_mut() = Some(Box::new(move || {
            rng.borrow_mut().nextu() as f64 * (1.0 / (u32::MAX as f64 + 1.0))
        }));
    });
    BITS_32.with(|c| {
        *c.borrow_mut() = Some(Box::new(move || rng2.borrow_mut().nextu() as c_ulong));
    });

    let name = CString::new(battery.name()).unwrap();
    let mut ugen = Unif01Gen {
        name:     name.as_ptr() as *mut _,
        get_u01:  Some(get_u01_32),
        get_bits: Some(get_bits_32),
        write:    Some(write_noop),
        param:    std::ptr::null_mut(),
        state:    std::ptr::null_mut(),
    };

    unsafe { run_battery(&raw mut ugen as *mut Unif01Gen, battery) };

    let result = collect(battery);
    clear_callbacks_32();
    result
}

/// Run a TestU01 battery on a [`Rng64`] generator.
///
/// TestU01 prints per-test results directly to stdout.
///
/// # Example
/// ```no_run
/// use urng::prelude::*;
/// use urng::testing::testu01::{self, Battery};
///
/// let rng = Xoshiro256Pp::new(42);
/// let result = testu01::run64(rng, Battery::SmallCrush);
/// assert!(result.passed(), "{} failed {}/{}", result.battery, result.n_failed, result.n_tests);
/// ```
pub fn run64<R: Rng64 + 'static>(rng: R, battery: Battery) -> TestResult {
    let _guard = LOCK.lock().expect("testu01 mutex poisoned");

    let rng = Rc::new(RefCell::new(rng));
    let rng2 = Rc::clone(&rng);
    U01_64.with(|c| {
        *c.borrow_mut() = Some(Box::new(move || {
            rng.borrow_mut().nextu() as f64 * (1.0 / (u64::MAX as f64 + 1.0))
        }));
    });
    BITS_64.with(|c| {
        // c_ulong is 32-bit on Windows; return lower 32 bits.
        *c.borrow_mut() = Some(Box::new(move || rng2.borrow_mut().nextu() as c_ulong));
    });

    let name = CString::new(battery.name()).unwrap();
    let mut ugen = Unif01Gen {
        name:     name.as_ptr() as *mut _,
        get_u01:  Some(get_u01_64),
        get_bits: Some(get_bits_64),
        write:    Some(write_noop),
        param:    std::ptr::null_mut(),
        state:    std::ptr::null_mut(),
    };

    unsafe { run_battery(&raw mut ugen as *mut Unif01Gen, battery) };

    let result = collect(battery);
    clear_callbacks_64();
    result
}
