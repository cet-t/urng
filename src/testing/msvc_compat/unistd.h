/* Stub unistd.h for MSVC.
 * POSIX unistd.h is absent from MSVC's CRT.
 * Only gethostname() is needed by TestU01's gdef.c; it is available via
 * winsock2.h but including that here causes conflicts if windows.h was
 * already included. Declare it directly instead.
 */
#pragma once
#ifndef _UNISTD_H
#define _UNISTD_H

#ifndef _WINSOCKAPI_   /* winsock.h not yet included */
#  ifndef _WINSOCK2API_ /* winsock2.h not yet included */
int __cdecl gethostname(char *name, int namelen);
#  endif
#endif

#endif /* _UNISTD_H */
