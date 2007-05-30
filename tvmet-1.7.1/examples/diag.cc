/* Version: $Id: diag.cc,v 1.1 2003/02/12 19:03:48 opetzold Exp $ */


#include <iostream>
#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>


using namespace std;
using namespace tvmet;

typedef Matrix<double,3,3>	matrix33d;
typedef Vector<double,3>	vector3d;


int main()
{
  matrix33d m1, m2(0);
  vector3d v1, v2;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  v1 = diag(m1);

  // not yet, since we need to assign an expression/scalar to an expression
  // diag(m2) = 1.0;

  cout << "M1 = " << m1 << endl;
  cout << "diag(M1) = " << v1 << endl;
  cout << "identity(M2) = " << m2 << endl;
}
