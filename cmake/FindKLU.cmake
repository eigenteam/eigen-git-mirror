# KLU lib usually requires linking to a blas library.
# It is up to the user of this module to find a BLAS and link to it.

if (KLU_INCLUDES AND KLU_LIBRARIES)
  set(KLU_FIND_QUIETLY TRUE)
endif (KLU_INCLUDES AND KLU_LIBRARIES)

find_path(KLU_INCLUDES
  NAMES
  klu.h
  PATHS
  $ENV{KLUDIR}
  ${INCLUDE_INSTALL_DIR}
  PATH_SUFFIXES
  suitesparse
  ufsparse
)

if(KLU_LIBRARIES)

  if(NOT KLU_LIBDIR)
    get_filename_component(KLU_LIBDIR ${KLU_LIBRARIES} PATH)
  endif(NOT KLU_LIBDIR)

  find_library(COLAMD_LIBRARY colamd PATHS ${KLU_LIBDIR} $ENV{KLUDIR} ${LIB_INSTALL_DIR})
  if(COLAMD_LIBRARY)
    set(KLU_LIBRARIES ${KLU_LIBRARIES} ${COLAMD_LIBRARY})
  endif ()
  
  find_library(AMD_LIBRARY amd PATHS ${KLU_LIBDIR} $ENV{KLUDIR} ${LIB_INSTALL_DIR})
  if(AMD_LIBRARY)
    set(KLU_LIBRARIES ${KLU_LIBRARIES} ${AMD_LIBRARY})
  endif ()

  find_library(SUITESPARSE_LIBRARY SuiteSparse PATHS ${KLU_LIBDIR} $ENV{KLUDIR} ${LIB_INSTALL_DIR})
  if(SUITESPARSE_LIBRARY)
    set(KLU_LIBRARIES ${KLU_LIBRARIES} ${SUITESPARSE_LIBRARY})
  endif ()

  find_library(CHOLMOD_LIBRARY cholmod PATHS $ENV{KLU_LIBDIR} $ENV{KLUDIR} ${LIB_INSTALL_DIR})
  if(CHOLMOD_LIBRARY)
    set(KLU_LIBRARIES ${KLU_LIBRARIES} ${CHOLMOD_LIBRARY})
  endif()

endif(KLU_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(KLU DEFAULT_MSG
                                  KLU_INCLUDES KLU_LIBRARIES)

mark_as_advanced(KLU_INCLUDES KLU_LIBRARIES AMD_LIBRARY COLAMD_LIBRARY CHOLMOD_LIBRARY SUITESPARSE_LIBRARY)
