#include <Eigen/Dense>
#include <iostream>

int main()
{
  Eigen::MatrixXf m(3,3);
  m << 1,2,3,
       4,5,6,
       7,8,9;
  std::cout << "2nd Row: " << m.row(1) << std::endl;
  m.col(0) += m.col(2);
  std::cout << "m after adding third column to first:\n";
  std::cout << m << std::endl;
}
