
if (TAUCS_INCLUDES AND TAUCS_LIBRARIES)
  set(TAUCS_FIND_QUIETLY TRUE)
endif (TAUCS_INCLUDES AND TAUCS_LIBRARIES)

find_path(TAUCS_INCLUDES
  NAMES
  taucs.h
  PATHS
  $ENV{TAUCSDIR}
  ${INCLUDE_INSTALL_DIR}
)

find_library(TAUCS_LIBRARIES taucs PATHS $ENV{TAUCSDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TAUCS DEFAULT_MSG
                                  TAUCS_INCLUDES TAUCS_LIBRARIES)

mark_as_advanced(TAUCS_INCLUDES TAUCS_LIBRARIES)
