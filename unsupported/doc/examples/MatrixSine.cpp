#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

int main()
{
  MatrixXd A = MatrixXd::Random(3,3);
  std::cout << "A = \n" << A << "\n\n";

  MatrixXd sinA = ei_matrix_sin(A);
  std::cout << "sin(A) = \n" << sinA << "\n\n";

  MatrixXd cosA = ei_matrix_cos(A);
  std::cout << "cos(A) = \n" << cosA << "\n\n";
  
  // The matrix functions satisfy sin^2(A) + cos^2(A) = I, 
  // like the scalar functions.
  std::cout << "sin^2(A) + cos^2(A) = \n" << sinA*sinA + cosA*cosA << "\n\n";
}
