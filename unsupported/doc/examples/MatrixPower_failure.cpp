#include <unsupported/Eigen/MatrixFunctions>
#include <iostream>

int main()
{
  Eigen::Matrix4d A;
  A << 0, 0, 2, 3,
       0, 0, 4, 5,
       0, 0, 6, 7,
       0, 0, 8, 9;
  std::cout << A.pow(0.37) << std::endl;
  
  // The 1 makes eigenvalue 0 non-semisimple.
  A.coeffRef(0, 1) = 1;

  // This fails if EIGEN_NO_DEBUG is undefined.
  std::cout << A.pow(0.37) << std::endl;

  return 0;
}
