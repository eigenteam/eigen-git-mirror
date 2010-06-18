#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

int main(int, char *[])
{
  Matrix3d m;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      m(row, column) = row + 0.01 * column;
  Vector3d v = Vector3d::Ones();
  std::cout << m * v << std::endl;
}
