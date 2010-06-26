#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
int main()
{
  Matrix2d mat;
  mat << 1, 2,
         3, 4;
  Vector2d vec(-1,1);
  RowVector2d rowvec(2,0);
  std::cout << "Here is mat*mat:\n" << mat*mat << std::endl;
  std::cout << "Here is mat*vec:\n" << mat*vec << std::endl;
  std::cout << "Here is rowvec*mat:\n" << rowvec*mat << std::endl;
  std::cout << "Here is rowvec*vec:\n" << rowvec*vec << std::endl;
  std::cout << "Here is vec*rowvec:\n" << vec*rowvec << std::endl;
  std::cout << "Let's multiply mat by itself" << std::endl;
  std::cout << "Now mat is mat:\n" << mat << std::endl;
}
