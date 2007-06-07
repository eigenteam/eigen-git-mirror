INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_LONG_DOUBLE _RESULT)

SET(_CHECK_LONG_DOUBLE_SOURCE_CODE "

int main(int argc, char *argv[])
{
  long double ld = static_cast<long double>(0);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_LONG_DOUBLE_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_LONG_DOUBLE)
