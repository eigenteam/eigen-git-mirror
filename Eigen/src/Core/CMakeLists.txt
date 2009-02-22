FILE(GLOB Eigen_Core_SRCS "*.h")

INSTALL(FILES
  ${Eigen_Core_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core
  )

FILE(GLOB Eigen_Core_Product_SRCS "products/*.h")

INSTALL(FILES
  ${Eigen_Core_Product_SRCS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen/src/Core/products
  )


ADD_SUBDIRECTORY(util)
ADD_SUBDIRECTORY(arch)
