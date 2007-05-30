dnl $Id: ac_cxx_have_sysv_math.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check for compiler support for double SYSV math, if there define
dnl HAVE_SYSV_MATH.
dnl

AC_DEFUN([AC_CXX_HAVE_SYSV_MATH],
[AC_CACHE_CHECK(for SYSV math library,
ac_cv_cxx_have_sysv_math,
[AC_LANG_PUSH([C++])
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_LINK_IFELSE([AC_LANG_PROGRAM([[
#ifndef _ALL_SOURCE
 #define _ALL_SOURCE
#endif
#ifndef _XOPEN_SOURCE
 #define _XOPEN_SOURCE
#endif
#ifndef _XOPEN_SOURCE_EXTENDED
 #define _XOPEN_SOURCE_EXTENDED 1
#endif
#include <cmath>]], [[double x = 1.0; double y = 1.0;
_class(x);
ilogb(x);
itrunc(x);
nearest(x);
rsqrt(x);
uitrunc(x);

copysign(x,y);
drem(x,y);
fmod(x,y);
hypot(x,y);
nextafter(x,y);
remainder(x,y);
scalb(x,y);
unordered(x,y);
return 0;]])],[ac_cv_cxx_have_sysv_math=yes],[ac_cv_cxx_have_sysv_math=no])
 LIBS="$ac_save_LIBS"
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_sysv_math" = yes; then
  AC_DEFINE(HAVE_SYSV_MATH,,[Define if the compiler supports SYSV math library])
fi
])
