set(Eigen_HEADERS Core LU Cholesky QR Geometry Sparse Array SVD LeastSquares QtAlignedMalloc StdVector Householder Jacobi Eigenvalues)

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/Eigen COMPONENT Devel
  )

add_subdirectory(src)
