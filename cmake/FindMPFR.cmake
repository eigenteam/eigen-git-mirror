# Try to find the MPFR library
# See http://www.mpfr.org/

if (MPFR_INCLUDES AND MPFR_LIBRARIES)
  set(MPFR_FIND_QUIETLY TRUE)
endif (MPFR_INCLUDES AND MPFR_LIBRARIES)

find_path(MPFR_INCLUDES
  NAMES
  mpfr.h
  PATHS
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
)

find_library(MPFR_LIBRARIES mpfr PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFR DEFAULT_MSG
                                  MPFR_INCLUDES MPFR_LIBRARIES)
mark_as_advanced(MPFR_INCLUDES MPFR_LIBRARIES)
