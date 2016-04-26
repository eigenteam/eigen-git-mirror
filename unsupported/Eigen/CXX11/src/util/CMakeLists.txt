FILE(GLOB Eigen_CXX11_util_SRCS "*.h")

INSTALL(FILES
  ${Eigen_CXX11_util_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11/src/util COMPONENT Devel
  )
