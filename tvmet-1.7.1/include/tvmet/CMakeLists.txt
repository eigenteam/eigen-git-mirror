INCLUDE (CheckIncludeFiles)
INCLUDE (CheckComplex)
INCLUDE (CheckComplexMath1)
INCLUDE (CheckComplexMath2)
INCLUDE (CheckIeeeMath)
INCLUDE (CheckSysvMath)

# the following are directories where stuff will be installed to
set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/tvmet"
    CACHE PATH
    "The subdirectory to the header prefix"
    FORCE)

FILE(GLOB tvmet_header_SRCS "*.h")

CHECK_INCLUDE_FILES (dlfcn.h TVMET_HAVE_DLFCN_H)
CHECK_COMPLEX       (TVMET_HAVE_COMPLEX)
CHECK_COMPLEX_MATH1 (TVMET_HAVE_COMPLEX_MATH1)
CHECK_COMPLEX_MATH2 (TVMET_HAVE_COMPLEX_MATH2)
CHECK_IEEE_MATH     (TVMET_HAVE_IEEE_MATH)
CHECK_SYSV_MATH     (TVMET_HAVE_SYSV_MATH)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
               ${INCLUDE_INSTALL_DIR}/config.h)

install(FILES 
  ${tvmet_header_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  )

