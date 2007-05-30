#include <iostream>
#include <iomanip>
#include <algorithm>

#include <cstdlib>

#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>

using namespace std;
using namespace tvmet;

typedef Vector<double,3>	vector3d;
typedef Matrix<double,3,3>	matrix33d;

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

  std::generate(v1.begin(), v1.end(), std::rand);
  std::generate(m1.begin(), m1.end(), std::rand);

  testMV(vr, m1, v1);

  cout << std::setw(12) << m1 << " * " << std::setw(12) << v1 << " =\n";
  cout << std::setw(12) << vr << endl;
}
