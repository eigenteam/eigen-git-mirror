#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
  MatrixXd m(3,3);
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      m(row, column) = row + 0.01 * column;
  VectorXd v = VectorXd::Ones(3);
  std::cout << m * v << std::endl;
}
