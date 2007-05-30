dnl $Id: ac_cxx_have_ieee_math.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check for compiler support for double IEEE math, if there define
dnl HAVE_IEEE_MATH.
dnl

AC_DEFUN([AC_CXX_HAVE_IEEE_MATH],
[AC_CACHE_CHECK(for IEEE math library,
ac_cv_cxx_have_ieee_math,
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
acosh(x); asinh(x); atanh(x);
expm1(x);
erf(x); erfc(x);
// finite(x);
isnan(x);
j0(x); j1(x);
lgamma(x);
logb(x); log1p(x);
rint(x);
// trunc(x);
y0(x); y1(x);
return 0;]])],[ac_cv_cxx_have_ieee_math=yes],[ac_cv_cxx_have_ieee_math=no])
 LIBS="$ac_save_LIBS"
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_ieee_math" = yes; then
  AC_DEFINE(HAVE_IEEE_MATH,,[Define if the compiler supports IEEE math library])
fi
])
