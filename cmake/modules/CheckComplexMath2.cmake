INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_COMPLEX_MATH2 _RESULT)

SET(_CHECK_COMPLEX_MATH2_SOURCE_CODE "

#include<complex>

using namespace std;

int main(int argc, char *argv[])
{
  complex<double> x(1.0, 1.0), y(1.0, 1.0);
  acos(x); asin(x); atan(x); atan2(x,y); atan2(x, double(3.0));
  atan2(double(3.0), x); log10(x); return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_COMPLEX_MATH2_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_COMPLEX_MATH2)
