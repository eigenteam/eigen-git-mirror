#include <Eigen/Dense>
#include <iostream>

int main()
{
  Eigen::MatrixXf m(4,4);
  m <<  1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16;
  std::cout << "Block in the middle" << std::endl;
  std::cout << m.block<2,2>(1,1) << std::endl << std::endl;
  for (int i = 1; i < 4; ++i) 
  {
    std::cout << "Block of size " << i << std::endl;
    std::cout << m.block(0,0,i,i) << std::endl << std::endl;
  }
}
