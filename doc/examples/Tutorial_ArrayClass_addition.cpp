#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  ArrayXXf a(3,3);
  ArrayXXf b(3,3);
  
  a << 1,2,3,
       4,5,6,
       7,8,9;

  b << 1,2,3,
       1,2,3,
       1,2,3;
       
  cout << "a + b = " << endl
       << a + b << endl;
}
