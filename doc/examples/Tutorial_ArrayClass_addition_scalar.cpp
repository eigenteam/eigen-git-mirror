#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

int main()
{
  ArrayXXf a(3,3);
  
  a << 1,2,3,
       4,5,6,
       7,8,9;

  cout << "a + 2 = " << endl
       << a + 2 << endl;
}
