dnl $Id: ac_prog_cxx_intel.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl Check for Intel C++ compiler
dnl

AC_DEFUN([AC_PROG_CXX_INTEL],
[AC_CACHE_CHECK(whether we are using INTEL C++, INTEL_CXX,
 [cat > conftest.c <<EOF
# if defined(__ICL) || defined(__ICC)
  yes;
#endif
EOF
if AC_TRY_COMMAND(${CXX} -E conftest.c) | egrep yes >/dev/null 2>&1; then
  INTEL_CXX=yes
  compiler=intelcc
else
  INTEL_CXX=no
fi])])
