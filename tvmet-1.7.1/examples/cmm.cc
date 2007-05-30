#include <iostream>
#include <complex>

#include <tvmet/Matrix.h>

using namespace tvmet;
using std::cout;
using std::endl;

typedef Matrix<std::complex<double>,3,3>	matrix33d;

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
