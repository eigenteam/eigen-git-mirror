dnl $Id: ac_c_long_long.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Provides a test for the existance of the long long int type and
dnl defines HAVE_LONG_LONG if it is found.
dnl

AC_DEFUN([AC_C_LONG_LONG],
[AC_CACHE_CHECK(for long long int, ac_cv_c_long_long,
[AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
   long long int i;
]], [[]])],[ac_cv_c_long_long=yes],[ac_cv_c_long_long=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_c_long_long" = yes; then
  AC_DEFINE(HAVE_LONG_LONG,,[Define if the compiler supports the long_long type])
fi
])
