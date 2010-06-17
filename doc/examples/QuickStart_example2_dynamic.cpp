#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int, char *[])
{
  MatrixXd m(3,3);
  for (int rowIndex = 0; rowIndex < 3; ++rowIndex)
    for (int columnIndex = 0; columnIndex < 3; ++columnIndex)
      m(rowIndex, columnIndex) = rowIndex + 0.01 * columnIndex;
  VectorXd v = VectorXd::Ones(3);
  std::cout << m * v << std::endl;
}
