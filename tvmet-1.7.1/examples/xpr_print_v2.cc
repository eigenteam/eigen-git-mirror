#include <iostream>
#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

int main() {
  // Vector stuff II (unary functions)
  Vector<double,3>	v1(1,2,3), v2(v1);
  Vector<double,3>	v3(0);

  cout << "Xpr Level printing of "
       << "sqrt(" << v1 <<"):\n";
  cout << sqrt(v1) << endl;
  cout << "The result = \n";
  v3 = sqrt(v1);
  cout << v3 << endl << endl;
}

