#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;

int main()
{
  MatrixXf m(3,3);
  
  m << 1,2,3,
       4,5,6,
       7,8,9;
     
  std::cout << "2nd Row: "
    << m.row(1) << std::endl;
}
