FILE(GLOB Eigen_SpecialFunctions_SRCS "*.h")
INSTALL(FILES
  ${Eigen_SpecialFunctions_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/SpecialFunctions COMPONENT Devel
  )

FILE(GLOB Eigen_SpecialFunctions_arch_CUDA_SRCS "arch/CUDA/*.h")
INSTALL(FILES
  ${Eigen_SpecialFunctions_arch_CUDA_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen/src/SpecialFunctions/arch/CUDA COMPONENT Devel
  )