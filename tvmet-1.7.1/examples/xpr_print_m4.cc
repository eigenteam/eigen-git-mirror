#include <iostream>
#include <tvmet/Matrix.h>

using namespace std;
using namespace tvmet;

int main() {
  // Matrix stuff IV (binary functions with pod)
  Matrix<double,3,3>	m1, m2, m3;
  m1 = 1,2,3,
       4,5,6,
       7,8,9;
  m2 = trans(m1);

  cout << "Xpr Level printing of "
       << "pow(\n" << m1 << ", " << 3 << "):\n";
  cout << pow(m1, 3) << endl;
  cout << "The result = \n";
  m3 = pow(m1, 3);
  cout << m3 << endl << endl;
}
