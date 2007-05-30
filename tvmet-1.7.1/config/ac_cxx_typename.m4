dnl $Id: ac_cxx_typename.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check compiler support for "typename" keyword, define HAVE_TYPENAME if there.
dnl

AC_DEFUN([AC_CXX_TYPENAME],
[AC_CACHE_CHECK(for typename,
ac_cv_cxx_typename,
[ AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[template<typename T>class X {public:X(){}};]], [[X<float> x; return 0;]])],[ac_cv_cxx_typename=yes],[ac_cv_cxx_typename=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_typename" = yes; then
  AC_DEFINE(HAVE_TYPENAME,,[Define if the compiler recognizes typename])
fi
])
