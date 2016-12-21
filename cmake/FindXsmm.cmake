# libxsmm

if (XSMM_INCLUDES AND XSMM_LIBRARIES)
  set(XSMM_FIND_QUIETLY TRUE)
endif (XSMM_INCLUDES AND XSMM_LIBRARIES)

find_path(XSMM_INCLUDES 
  NAMES 
  scotch.h 
  PATHS 
  $ENV{XSMMDIR} 
  ${INCLUDE_INSTALL_DIR} 
  PATH_SUFFIXES 
  scotch
)


find_library(XSMM_LIBRARIES xsmm PATHS $ENV{XSMMDIR} ${LIB_INSTALL_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XSMM DEFAULT_MSG
                                  XSMM_INCLUDES XSMM_LIBRARIES)

mark_as_advanced(XSMM_INCLUDES XSMM_LIBRARIES)
