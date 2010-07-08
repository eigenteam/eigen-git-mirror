#include <Eigen/Dense>
#include <iostream>

using namespace std;
using namespace Eigen;

int main()
{
  VectorXf v(2);
  MatrixXf m(2,2), n(2,2);
  
  v << 5,
       10;
  
  m << 2,2,
       3,4;

  n << 1, 2,
       32,12;
  
  cout << "v.norm() = " << v.norm() << endl;
  cout << "m.norm() = " << m.norm() << endl;
  cout << "n.norm() = " << n.norm() << endl;
  cout << endl;
  cout << "v.squaredNorm() = " << v.squaredNorm() << endl;
  cout << "m.squaredNorm() = " << m.squaredNorm() << endl;
  cout << "n.squaredNorm() = " << n.squaredNorm() << endl;
}
