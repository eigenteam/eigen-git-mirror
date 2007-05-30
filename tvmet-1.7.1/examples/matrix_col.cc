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

int main()
{
  matrix33d 			m1;

  std::generate(m1.begin(), m1.end(), std::rand);

  vector3d vc0( col(m1, 0) );
  vector3d vc1( col(m1, 1) );
  vector3d vc2( col(m1, 2) );

  vector3d vr0( row(m1, 0) );
  vector3d vr1( row(m1, 1) );
  vector3d vr2( row(m1, 2) );


  cout << std::setw(12) << m1 << endl;

  cout << "col vectors:" << endl;
  cout << vc0 << endl;
  cout << vc1 << endl;
  cout << vc2 << endl;

  cout << "row vectors:" << endl;
  cout << vr0 << endl;
  cout << vr1 << endl;
  cout << vr2 << endl;

}

