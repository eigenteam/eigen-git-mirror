dnl $Id: ac_cxx_partial_specialization.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check if the compiler supports partial template specialization,
dnl on support define HAVE_PARTIAL_SPECIALIZATION.
dnl

AC_DEFUN([AC_CXX_PARTIAL_SPECIALIZATION],
[AC_CACHE_CHECK(whether the compiler supports partial specialization,
ac_cv_cxx_partial_specialization,
[AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
template<class T, int N> class X            { public : enum { e = 0 }; };
template<int N>          class X<double, N> { public : enum { e = 1 }; };
template<class T>        class X<T, 2>      { public : enum { e = 2 }; };
]], [[return (X<int,3>::e == 0) && (X<double,3>::e == 1) && (X<float,2>::e == 2);]])],[ac_cv_cxx_partial_specialization=yes],[ac_cv_cxx_partial_specialization=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_partial_specialization" = yes; then
  AC_DEFINE(HAVE_PARTIAL_SPECIALIZATION,,[Define if the compiler supports partial specialization])
fi
])
