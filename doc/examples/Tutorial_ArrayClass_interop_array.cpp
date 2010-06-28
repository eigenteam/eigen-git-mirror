#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  ArrayXXf m(2,2);
  ArrayXXf n(2,2);
  
  ArrayXXf result(2,2);

  //initialize arrays
  m << 1,2,
       3,4;

  n << 5,6,
       7,8;

  
  // --> array multiplication
  result = m * n;

  cout << "-- Array m*n: --" << endl
    << result << endl << endl;


  // --> Matrix multiplication
  result = m.matrix() * n.matrix();
  
  cout << "-- Matrix m*n: --" << endl
    << result << endl << endl;
}
