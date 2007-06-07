INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_COMPLEX _RESULT)

SET(_CHECK_COMPLEX_SOURCE_CODE "

#include<complex>

using namespace std;

int main(int argc, char *argv[])
{
  complex<double> x(1.0, 1.0);
  complex<float> y(1.0f, 1.0f);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_COMPLEX_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_COMPLEX)
