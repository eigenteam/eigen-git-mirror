dnl $Id: ac_cxx_have_complex_math2.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Available from the GNU Autoconf Macro Archive at:
dnl http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_complex_math2.html
dnl

AC_DEFUN([AC_CXX_HAVE_COMPLEX_MATH2],
[AC_CACHE_CHECK(whether the compiler has more complex math functions,
ac_cv_cxx_have_complex_math2,
[AC_REQUIRE([AC_CXX_NAMESPACES])
  AC_LANG_PUSH([C++])
 ac_save_LIBS="$LIBS"
 LIBS="$LIBS -lm"
 AC_LINK_IFELSE([AC_LANG_PROGRAM([[#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[complex<double> x(1.0, 1.0), y(1.0, 1.0);
acos(x); asin(x); atan(x); atan2(x,y); atan2(x, double(3.0));
atan2(double(3.0), x); log10(x); return 0;]])],[ac_cv_cxx_have_complex_math2=yes],[ac_cv_cxx_have_complex_math2=no])
 LIBS="$ac_save_LIBS"
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_complex_math2" = yes; then
  AC_DEFINE(HAVE_COMPLEX_MATH2,,[define if the compiler has more complex math functions])
fi
])
