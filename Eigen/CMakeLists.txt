FILE(GLOB Eigen_SRCS "*.h")

SET(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
    FORCE)

INSTALL(FILES 
  ${Eigen_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen
  )

ADD_SUBDIRECTORY(src)