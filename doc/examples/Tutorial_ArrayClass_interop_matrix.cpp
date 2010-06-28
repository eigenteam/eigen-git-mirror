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

  
  // --> matrix multiplication
  result = m * n;

  cout << "-- Matrix m*n: --" << endl
    << result << endl << endl;


  // --> coeff-wise multiplication
  result = m.array() * n.array();
  
  cout << "-- Array m*n: --" << endl
    << result << endl << endl;
  
  
  // ->> coeff-wise addition of a scalar
  result = m.array() + 4;
  
  cout << "-- Array m + 4: --" << endl
    << result << endl << endl;
}
