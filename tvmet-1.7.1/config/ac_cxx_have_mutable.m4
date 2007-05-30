dnl $Id: ac_cxx_have_mutable.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl If the compiler allows modifying data members of classes flagged with
dnl the mutable keyword even in const objects, define HAVE_MUTABLE.
dnl

AC_DEFUN([AC_CXX_HAVE_MUTABLE],
[AC_CACHE_CHECK(for mutable,
ac_cv_cxx_mutable,
[AC_LANG_PUSH([C++])
 AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
class Foo {
  mutable int i;
public:
  int bar (int n) const { i = n; return i; }
};
]], [[Foo foo; return foo.bar(1);]])],[ac_cv_cxx_mutable=yes],[ac_cv_cxx_mutable=no])
 AC_LANG_POP([C++])
])
if test "$ac_cv_cxx_mutable" = yes; then
  AC_DEFINE(HAVE_MUTABLE,,[Define if the compiler supports the mutable keyword])
fi
])
