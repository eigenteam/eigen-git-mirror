INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_COMPLEX_MATH1 _RESULT)

SET(_CHECK_COMPLEX_MATH1_SOURCE_CODE "

#include<complex>

using namespace std;

int main(int argc, char *argv[])
{
  complex<double> x(1.0, 1.0), y(1.0, 1.0);
  cos(x); cosh(x); exp(x); log(x); pow(x,1); pow(x,double(2.0));
  pow(x, y); pow(double(2.0), x); sin(x); sinh(x); sqrt(x); tan(x); tanh(x);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_COMPLEX_MATH1_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_COMPLEX_MATH1)
