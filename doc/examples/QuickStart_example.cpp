#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main(int, char *[])
{
  MatrixXd m(2,3);
  m(0,0) = -3;
  m(1,0) = 1.5;
  m.rightCols(2) = MatrixXd::Identity(2,2);
  std::cout << 2*m << std::endl;
}
