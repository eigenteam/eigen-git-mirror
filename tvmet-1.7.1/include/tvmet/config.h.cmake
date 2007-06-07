#ifndef _INCLUDE_TVMET_CONFIG_H
#define _INCLUDE_TVMET_CONFIG_H

/* define if the compiler has complex<T> */
#cmakedefine TVMET_HAVE_COMPLEX

/* define if the compiler has complex math functions */
#cmakedefine TVMET_HAVE_COMPLEX_MATH1

/* define if the compiler has more complex math functions */
#cmakedefine TVMET_HAVE_COMPLEX_MATH2

/* Define if the compiler supports IEEE math library */
#cmakedefine TVMET_HAVE_IEEE_MATH

/* Define to 1 if the long double type is supported and the
 * standard math library provides math functions for this type
 */
#cmakedefine TVMET_HAVE_LONG_DOUBLE  1 

/* Define if the compiler supports the long_long type */
#cmakedefine TVMET_HAVE_LONG_LONG

/* Define if the compiler supports SYSV math library */
#cmakedefine TVMET_HAVE_SYSV_MATH

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine TVMET_HAVE_SYS_TIME_H  1 

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine TVMET_HAVE_UNISTD_H  1

#define _tvmet_restrict @TVMET_RESTRICT_KEYWORD@

#define _tvmet_always_inline @TVMET_ALWAYS_INLINE@

/* _INCLUDE_TVMET_CONFIG_H */
#endif
