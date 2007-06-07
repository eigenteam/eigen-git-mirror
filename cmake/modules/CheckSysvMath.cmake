INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_SYSV_MATH _RESULT)

SET(_CHECK_SYSV_MATH_SOURCE_CODE "

#include<cmath>

int main(int argc, char *argv[])
{
  double x = 1.0; double y = 1.0;
  _class(x);
  ilogb(x);
  itrunc(x);
  nearest(x);
  rsqrt(x);
  uitrunc(x);
  
  copysign(x,y);
  drem(x,y);
  fmod(x,y);
  hypot(x,y);
  nextafter(x,y);
  remainder(x,y);
  scalb(x,y);
  unordered(x,y);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_SYSV_MATH_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_SYSV_MATH)
