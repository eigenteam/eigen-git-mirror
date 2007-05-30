dnl $Id: ac_cxx_have_namespaces.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl If the compiler can prevent names clashes using namespaces, define
dnl HAVE_NAMESPACES.
dnl

AC_DEFUN([AC_CXX_NAMESPACES],
[AC_CACHE_CHECK(for namespaces support,
ac_cv_cxx_namespaces,
[AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[namespace Outer { namespace Inner { int i = 0; }}]], [[using namespace Outer::Inner; return i;]])],[ac_cv_cxx_namespaces=yes],[ac_cv_cxx_namespaces=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_namespaces" = yes; then
  AC_DEFINE(HAVE_NAMESPACES,,[Define if the compiler implements namespaces])
fi
])
