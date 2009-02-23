FILE(GLOB Eigen_Core_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core COMPONENT Devel
  )

FILE(GLOB Eigen_Core_Product_SRCS "products/*.h")

INSTALL(FILES
  ${Eigen_Core_Product_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/products COMPONENT Devel
  )


ADD_SUBDIRECTORY(util)
ADD_SUBDIRECTORY(arch)
