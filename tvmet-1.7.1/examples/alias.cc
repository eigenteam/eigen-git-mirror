/*
 * $Id: alias.cc,v 1.1 2004/03/26 07:56:32 opetzold Exp $
 *
 * This example shows the solution of the problem with aliasing
 * mentioned at
 * http://tvmet.sourceforge.net/notes.html#alias
 */

#include <iostream>
#include <algorithm>
#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>
#include <tvmet/util/Incrementor.h>

using namespace std;


int main()
{
  typedef tvmet::Matrix<double, 3, 3>		matrix_type;

  matrix_type						M;

  std::generate(M.begin(), M.end(),
		tvmet::util::Incrementor<matrix_type::value_type>());

  std::cout << "M = " << M << std::endl;

  alias(M) = M * trans(M);

  std::cout << M << std::endl;

}
