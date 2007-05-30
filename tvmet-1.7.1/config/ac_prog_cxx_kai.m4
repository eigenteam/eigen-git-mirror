dnl $Id: ac_prog_cxx_kai.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check for KAI C++ compiler
dnl

AC_DEFUN([AC_PROG_CXX_KAI],
[AC_CACHE_CHECK(whether we are using KAI C++,
 KAI_CXX,
 [cat > conftest.c <<EOF
# if defined(__KCC)
  yes;
#endif
EOF
 if AC_TRY_COMMAND(${CXX} -E conftest.c) | egrep yes >/dev/null 2>&1; then
  KAI_CXX=yes
  compiler=kaicc
 else
  KAI_CXX=no
fi])])
