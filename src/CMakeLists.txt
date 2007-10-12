FILE(GLOB Eigen_SRCS "*.h")

SET(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
    FORCE)

INSTALL(FILES 
  ${gen_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  )

add_subdirectory(Core)