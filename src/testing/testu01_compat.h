/* MSVC compatibility shim for TestU01.
 * Provides sincos() which is a POSIX/GNU extension not present in MSVC's CRT.
 * Force-included via /FI when compiling with cl.exe.
 */
#if defined(_MSC_VER) && !defined(_TESTU01_COMPAT_H)
#define _TESTU01_COMPAT_H
#include <math.h>
static __inline void sincos(double x, double *s, double *c) {
    *s = sin(x);
    *c = cos(x);
}
#endif
