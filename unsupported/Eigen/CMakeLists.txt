set(Eigen_HEADERS AdolcForward BVH IterativeSolvers MatrixFunctions MoreVectorization AutoDiff AlignedVector3 Polynomials)

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/unsupported/Eigen COMPONENT Devel
  )

add_subdirectory(src)
