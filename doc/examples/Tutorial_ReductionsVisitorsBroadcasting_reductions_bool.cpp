#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  MatrixXf m(2,2), n(2,2);
  
  m << 0,2,
       3,4;

  n << 1,2,
       3,4;
  
  cout << "m.all()   = " << m.all() << endl;
  cout << "m.any()   = " << m.any() << endl;
  cout << "m.count() = " << m.count() << endl;
  cout << endl;
  cout << "n.all()   = " << n.all() << endl;
  cout << "n.any()   = " << n.any() << endl;
  cout << "n.count() = " << n.count() << endl;
}
