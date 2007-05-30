#ifndef _INCLUDE_TVMET_CONFIG_H
#define _INCLUDE_TVMET_CONFIG_H

/* define if the compiler has complex<T> */
#cmakedefine TVMET_HAVE_COMPLEX

/* define if the compiler has complex math functions */
#cmakedefine TVMET_HAVE_COMPLEX_MATH1

/* define if the compiler has more complex math functions */
#cmakedefine TVMET_HAVE_COMPLEX_MATH2

/* Define to 1 if you have the <dlfcn.h> header file. */
#cmakedefine TVMET_HAVE_DLFCN_H 1

/* Define to 1 if you have the `floor' function. */
#cmakedefine TVMET_HAVE_FLOOR  1

/* Define if the compiler supports IEEE math library */
#cmakedefine TVMET_HAVE_IEEE_MATH

/* Define to 1 if you have the <inttypes.h> header file. */
#cmakedefine TVMET_HAVE_INTTYPES_H  1 

/* Define to 1 if you have the `dl' library (-ldl). */
#cmakedefine TVMET_HAVE_LIBDL  1 

/* Define to 1 if long double works and has more range or precision than
   double. */
#cmakedefine TVMET_HAVE_LONG_DOUBLE  1 

/* Define if the compiler supports the long_long type */
#cmakedefine TVMET_HAVE_LONG_LONG

/* Define to 1 if you have the <memory.h> header file. */
#cmakedefine TVMET_HAVE_MEMORY_H  1 

/* Define to 1 if you have the `pow' function. */
#cmakedefine TVMET_HAVE_POW  1 

/* Define to 1 if you have the `rint' function. */
#cmakedefine TVMET_HAVE_RINT  1 

/* Define to 1 if you have the `sqrt' function. */
#cmakedefine TVMET_HAVE_SQRT  1 

/* Define if the compiler supports SYSV math library */
#cmakedefine TVMET_HAVE_SYSV_MATH

/* Define to 1 if you have the <sys/stat.h> header file. */
#cmakedefine TVMET_HAVE_SYS_STAT_H  1 

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine TVMET_HAVE_SYS_TIME_H  1 

/* Define to 1 if you have the <sys/types.h> header file. */
#cmakedefine TVMET_HAVE_SYS_TYPES_H  1 

/* Define if the compiler recognizes typename */
// ALWAYS ON -- so remove this define in the future.
#define TVMET_HAVE_TYPENAME

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine TVMET_HAVE_UNISTD_H  1

/* Define to 1 if you have the ANSI C header files. */
#cmakedefine TVMET_STDC_HEADERS  1

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
#cmakedefine TVMET_TM_IN_SYS_TIME 1

/* _INCLUDE_TVMET_CONFIG_H */
#endif
