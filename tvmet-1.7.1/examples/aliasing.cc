/*
 * $Id: aliasing.cc,v 1.2 2004/03/26 07:58:06 opetzold Exp $
 *
 * This example shows the problem with aliasing mentioned at
 * http://tvmet.sourceforge.net/notes.html#alias
 */

#include <iostream>
#include <algorithm>
#include <tvmet/Matrix.h>
#include <tvmet/util/Incrementor.h>

using std::cout; using std::endl;
using namespace tvmet;

typedef Matrix<double,3,3>	matrix_type;

int main()
{
  matrix_type 			A, B;
  matrix_type 			C;

  std::generate(A.begin(), A.end(),
		tvmet::util::Incrementor<matrix_type::value_type>());
  std::generate(B.begin(), B.end(),
		tvmet::util::Incrementor<matrix_type::value_type>());

  cout << "A = " << A << endl;
  cout << "B = " << B << endl;

  // matrix prod without aliasing
  C = A * B;
  cout << "C = A * B = " << C << endl;

  // work around for aliasing
  matrix_type 			temp_A(A);
  A = temp_A * B;
  cout << "matrix_type temp_A(A);\n"
       << "A = temp_A * B = " << A << endl;

  // this shows the aliasing problem
  A = A * B;
  cout << "A = A * B = " << A << endl;
}
