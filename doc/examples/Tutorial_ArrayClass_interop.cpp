#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  
  MatrixXf result(2,2);

  //initialize matrices
  m << 1,2,
       3,4;

  n << 5,6,
       7,8;
  
  // mix of array and matrix operations
  //   first coefficient-wise addition
  //   then the result is used with matrix multiplication
  result = (m.array() + 4).matrix() * m;

  cout << "-- Combination 1: --" << endl
    << result << endl << endl;


  // mix of array and matrix operations
  //   first coefficient-wise multiplication
  //   then the result is used with matrix multiplication
  result = (m.array() * n.array()).matrix() * m;

  cout << "-- Combination 2: --" << endl
    << result << endl << endl;

}
