/* sincos compatibility for MSVC.
 * POSIX sincos() is absent from MSVC's CRT; provide a portable fallback.
 * Compiled into the testu01 static library alongside the test batteries.
 */
#include <math.h>

void sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}
