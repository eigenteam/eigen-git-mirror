INCLUDE (CheckIncludeFiles)

# the following are directories where stuff will be installed to
set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/tvmet"
    CACHE PATH
    "The subdirectory to the header prefix"
    FORCE)

FILE(GLOB gmm_header_SRCS "*.h")

CHECK_INCLUDE_FILES (dlfcn.h TVMET_HAVE_DLFCN_H)
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake
               ${INCLUDE_INSTALL_DIR}/config.h)

install(FILES 
  ${gmm_header_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  )

