#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

int main(int, char *[])
{
  Matrix3d m;
  for (int rowIndex = 0; rowIndex < 3; ++rowIndex)
    for (int columnIndex = 0; columnIndex < 3; ++columnIndex)
      m(rowIndex, columnIndex) = rowIndex + 0.01 * columnIndex;
  Vector3d v = Vector3d::Ones();
  std::cout << m * v << std::endl;
}
