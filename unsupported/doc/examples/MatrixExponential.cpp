#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

int main()
{
  const double pi = std::acos(-1.0);

  MatrixXd A(3,3);
  A << 0,    -pi/4, 0,
       pi/4, 0,     0,
       0,    0,     0;
  std::cout << "The matrix A is:\n" << A << "\n\n";

  MatrixXd B;
  ei_matrix_exponential(A, &B);
  std::cout << "The matrix exponential of A is:\n" << B << "\n\n";
}
