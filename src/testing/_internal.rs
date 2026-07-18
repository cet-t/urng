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

/// Complementary error function `erfc(x)`, using the Abramowitz & Stegun
/// 7.1.26 rational approximation (max absolute error ~1.5e-7). Sufficient
/// precision for p-value computation in statistical randomness tests.
pub(crate) fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Standard normal CDF `Phi(x)`, derived from [`erfc`].
pub(crate) fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Upper regularized incomplete gamma function `Q(a, x) = Gamma(a, x) / Gamma(a)`,
/// used to derive chi-squared p-values: `p = igamc(df / 2, chi2 / 2)`.
///
/// Uses a series expansion for `x < a + 1` and a continued fraction for
/// `x >= a + 1`, following the standard Numerical-Recipes-style algorithm.
pub(crate) fn igamc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - igamma_series(a, x)
    } else {
        igamma_cf(a, x)
    }
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation.
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + G + 0.5;
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

fn igamma_series(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut sum = 1.0 / a;
    let mut del = sum;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * 1e-15 {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

fn igamma_cf(a: f64, x: f64) -> f64 {
    let gln = ln_gamma(a);
    let tiny = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}
