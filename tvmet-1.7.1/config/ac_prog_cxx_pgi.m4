dnl $Id: ac_prog_cxx_pgi.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Portland Group Incorporated C++ compiler
dnl

AC_DEFUN([AC_PROG_CXX_PGI],
[AC_CACHE_CHECK(whether we are using PGI C++, PGI_CXX,
 [cat > conftest.c <<EOF
# if defined(__PGI)
  yes;
#endif
EOF
if AC_TRY_COMMAND(${CXX} -E conftest.c) | egrep yes >/dev/null 2>&1; then
 PGI_CXX=yes
 compiler=pgicc
else
  PGI_CXX=no
fi])])
