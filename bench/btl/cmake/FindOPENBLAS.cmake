
if (OPENBLAS_LIBRARIES)
  set(OPENBLAS_FIND_QUIETLY TRUE)
endif (OPENBLAS_LIBRARIES)

find_file(OPENBLAS_LIBRARIES libopenblas.so PATHS /usr/lib $ENV{OPENBLASDIR} ${LIB_INSTALL_DIR})
find_library(OPENBLAS_LIBRARIES openblas PATHS $ENV{OPENBLASDIR} ${LIB_INSTALL_DIR})

if(OPENBLAS_LIBRARIES AND CMAKE_COMPILER_IS_GNUCXX)
  set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARIES} "-lpthread -lgfortran")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENBLAS DEFAULT_MSG
                                  OPENBLAS_LIBRARIES)

mark_as_advanced(OPENBLAS_LIBRARIES)
