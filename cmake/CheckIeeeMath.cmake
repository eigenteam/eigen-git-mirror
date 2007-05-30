INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_IEEE_MATH _RESULT)

SET(_CHECK_IEEE_MATH_SOURCE_CODE "

#include<cmath>

int main(int argc, char *argv[])
{
  double x = 1.0; double y = 1.0;
  acosh(x); asinh(x); atanh(x);
  expm1(x);
  erf(x); erfc(x);
  // finite(x);
  isnan(x);
  j0(x); j1(x);
  lgamma(x);
  logb(x); log1p(x);
  rint(x);
  // trunc(x);
  y0(x); y1(x);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_IEEE_MATH_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_IEEE_MATH)
