# This module checks if the C++ compiler supports the restrict keyword or
# some variant of it. The following variants are checked for in that order:
# 1. restrict            (The standard C99 keyword, not yet in C++ standard)
# 2. __restrict          (G++ has it)
# 3. __restrict__        (G++ has it too)
# 4. _Restrict           (seems to be used by Sun's compiler)
# These four cases seem to cover all existing variants; however some C++
# compilers don't support any variant, in which case the _RESULT variable is
# set to empty value.
#
# Copyright Benoit Jacob 2007 <jacob@math.jussieu.fr>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file in
# kdelibs/cmake/modules.

INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_RESTRICT_KEYWORD _RESULT)

SET(_CHECK_restrict_KEYWORD_SRC "

char f( const char * restrict x )
{
  return *x;
}

int main(int argc, char *argv[]) { return 0; } 
")

SET(_CHECK___restrict_KEYWORD_SRC "

char f( const char * __restrict x )
{
  return *x;
}

int main(int argc, char *argv[]) { return 0; } 
")

SET(_CHECK___restrict___KEYWORD_SRC "

char f( const char * __restrict__ x )
{
  return *x;
}

int main(int argc, char *argv[]) { return 0; } 
")

SET(_CHECK__Restrict_KEYWORD_SRC "

char f( const char * _Restrict x )
{
  return *x;
}

int main(int argc, char *argv[]) { return 0; } 
")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_restrict_KEYWORD_SRC}"
                          HAVE_KEYWORD_restrict)
IF(HAVE_KEYWORD_restrict)
  SET(${_RESULT} restrict)
ELSE(HAVE_KEYWORD_restrict)
  CHECK_CXX_SOURCE_COMPILES("${_CHECK___restrict_KEYWORD_SRC}"
                          HAVE_KEYWORD___restrict)
  IF(HAVE_KEYWORD___restrict)
    SET(${_RESULT} __restrict)
  ELSE(HAVE_KEYWORD___restrict)
    CHECK_CXX_SOURCE_COMPILES("${_CHECK___restrict___KEYWORD_SRC}"
                            HAVE_KEYWORD___restrict__)
    IF(HAVE_KEYWORD___restrict__)
      SET(${_RESULT} __restrict__)
    ELSE(HAVE_KEYWORD___restrict__)
      CHECK_CXX_SOURCE_COMPILES("${_CHECK__Restrict_KEYWORD_SRC}"
                            HAVE_KEYWORD__Restrict)
      IF(HAVE_KEYWORD__Restrict)
        SET(${_RESULT} _Restrict)
      ELSE(HAVE_KEYWORD__Restrict)
        SET(${_RESULT} ) # no variant of restrict keyword supported
      ENDIF(HAVE_KEYWORD__Restrict)
    ENDIF(HAVE_KEYWORD___restrict__)
  ENDIF(HAVE_KEYWORD___restrict)
ENDIF(HAVE_KEYWORD_restrict)

ENDMACRO (CHECK_RESTRICT_KEYWORD)
