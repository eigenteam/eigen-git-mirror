#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  Array33f m;
  m << 1,2,3,
       4,5,6,
       7,8,9;
  Array<float,5,5> n = Array<float,5,5>::Constant(0.6);
  n.block(1,1,3,3) = m;
  cout << "n = " << endl << n << endl << endl;
  Array33f res = n.block(0,0,3,3) * m;
  cout << "res =" << endl << res << endl;
}
