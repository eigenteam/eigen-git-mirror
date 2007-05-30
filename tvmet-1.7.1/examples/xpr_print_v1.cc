#include <iostream>
#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

int main() {
  // Vector stuff I
  Vector<double,3>	v1(1,2,3), v2(v1);
  Vector<double,3>	v3(0);

  cout << "Xpr Level printing of "
       << v1 << " * " << v2 <<":\n";
  cout << v1*v2 << endl;
  cout << "The result = \n";
  v3 = v1*v2;
  cout << v3 << endl << endl;
}

