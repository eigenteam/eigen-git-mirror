dnl $Id: ac_cxx_have_complex.m4,v 1.3 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Available from the GNU Autoconf Macro Archive at:
dnl http://www.gnu.org/software/ac-archive/htmldoc/ac_cxx_have_complex.html
dnl

AC_DEFUN([AC_CXX_HAVE_COMPLEX],
[AC_CACHE_CHECK(whether the compiler has complex<T>,
ac_cv_cxx_have_complex,
[AC_REQUIRE([AC_CXX_NAMESPACES])
 AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[#include <complex>
#ifdef HAVE_NAMESPACES
using namespace std;
#endif]], [[complex<float> a; complex<double> b; return 0;]])],[ac_cv_cxx_have_complex=yes],[ac_cv_cxx_have_complex=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_have_complex" = yes; then
  AC_DEFINE(HAVE_COMPLEX,,[define if the compiler has complex<T>])
fi
])
