SET(Eigen_HEADERS Core CoreDeclarations LU Cholesky QR)

SET(Eigen_SRCS
  src/Core/CoreInstanciations.cpp
)

ADD_LIBRARY(Eigen2 ${Eigen_SRCS})

SET(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen2"
    CACHE PATH
    "The directory where we install the header files"
    FORCE)

INSTALL(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen
  )

INSTALL(TARGETS Eigen2
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

ADD_SUBDIRECTORY(src)