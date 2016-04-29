FILE(GLOB Eigen_CXX11_ThreadPool_SRCS "*.h")

INSTALL(FILES
  ${Eigen_CXX11_ThreadPool_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/CXX11/src/ThreadPool COMPONENT Devel
  )
