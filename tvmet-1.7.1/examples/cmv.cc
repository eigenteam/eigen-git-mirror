#include <iostream>
#include <complex>

#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>

using namespace tvmet;
using std::cout;
using std::endl;

typedef Vector<std::complex<double>,3>		vector3d;
typedef Matrix<std::complex<double>,3,3>	matrix33d;

#if (defined __ICC )
#pragma warning(disable:1418) // external definition with no prior declaration
#endif

void testMV(vector3d& res, const matrix33d& m, const vector3d& v) {
  res = m * v;
}

int main()
{
  vector3d v1, vr;
  matrix33d m1;

  v1 = 1,2,3;
  m1 = 1,4,7,
       2,5,8,
       3,6,9;

  testMV(vr, m1, v1);

  cout << m1 << " * " << v1 << " =\n";
  cout << vr << endl;
}
