FILE(GLOB Eigen_Core_arch_ZVector_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_arch_ZVector_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/arch/ZVector COMPONENT Devel
)
