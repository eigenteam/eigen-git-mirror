# This module checks if the C++ compiler supports
# __attribute__((always_inline)).
#
# If yes, _RESULT is set to __attribute__((always_inline)).
# If no, _RESULT is set to empty value.
#
# Copyright Benoit Jacob 2007 <jacob@math.jussieu.fr>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file in
# kdelibs/cmake/modules.

INCLUDE(CheckCXXSourceCompiles)

MACRO (CHECK_ALWAYS_INLINE _RESULT)

SET(_CHECK_attribute_always_inline_SRC "

#ifdef __GNUC__
#  define ALL_IS_WELL
#endif

#ifdef __INTEL_COMPILER
#  if (__INTEL_COMPILER == 800) || (__INTEL_COMPILER > 800)
#    define ALL_IS_WELL
#  endif
#endif

#ifndef ALL_IS_WELL
#  error I guess your compiler doesn't support __attribute__((always_inline))
#endif

int main(int argc, char *argv[]) { return 0; } 
")

CHECK_CXX_SOURCE_COMPILES("${_CHECK_attribute_always_inline_SRC}"
                          HAVE_attribute_always_inline)
IF(HAVE_attribute_always_inline)
  SET(${_RESULT} "__attribute__((always_inline))")
ELSE(HAVE_attribute_always_inline)
  SET(${_RESULT} ) # attribute always_inline unsupported
ENDIF(HAVE_attribute_always_inline)

ENDMACRO (CHECK_ALWAYS_INLINE)
