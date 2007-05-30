#include <iostream>
#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>

using namespace std;
using namespace tvmet;

int main() {
 // Matrix Vector stuff II
  Vector<double,3>	v1(1,2,3), v2(v1);
  Vector<double,3>	v3(0);

  Matrix<double,3,3>	m1, m2, m3;
  m1 = 1,2,3,
       4,5,6,
       7,8,9;
  m2 = trans(m1);

  cout << "Xpr Level printing of "
       << "sqrt(\n" << m1 << "\n* " << v1 << ")\nresults into:\n";
  cout << sqrt(m1*v1) << endl;
  cout << "The result =\n";
  v3 = sqrt(m1*v1);
  cout << v3 << endl << endl;

}
