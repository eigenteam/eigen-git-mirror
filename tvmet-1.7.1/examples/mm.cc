#include <iostream>
#include <tvmet/Matrix.h>

using namespace std;
using namespace tvmet;

typedef Matrix<double,3,3>	matrix33d;

#if (defined __ICC )
#pragma warning(disable:1418) // external definition with no prior declaration
#endif

void testMM(matrix33d& res, const matrix33d& m1, const matrix33d& m2) {
  res = m1 * m2;
}

int main()
{
  matrix33d m1, m2, m3;

  m1 = 1,4,7,
       2,5,8,
       3,6,9;
  m2 = m1;

  testMM(m3, m1, m2);

  cout << m1 << "\n*\n" << m2 << "\n=";
  cout << m3 << endl;
}
