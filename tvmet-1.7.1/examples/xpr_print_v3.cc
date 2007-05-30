#include <iostream>
#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

int main() {
  // Vector stuff III (binary functions)
  Vector<double,3>	v1(1,2,3), v2(v1);
  Vector<double,3>	v3(0);

  cout << "Xpr Level printing of "
       << "pow(" << v1 << ", " << v2 << "):\n";
  cout << pow(v1, v2) << endl;
  cout << "The result = \n";
  v3 = pow(v1, v2);
  cout << v3 << endl << endl;
}

