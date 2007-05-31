INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_LONG_LONG _RESULT)

SET(_CHECK_LONG_LONG_SOURCE_CODE "

int main(int argc, char *argv[])
{
  long long ll = static_cast<long long>(0);
  long long int lli = static_cast<long long int>(0);
  return 0;
}

")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_LONG_LONG_SOURCE_CODE}" ${_RESULT})

ENDMACRO (CHECK_LONG_LONG)
