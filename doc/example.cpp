#include "../src/Core.h"

USING_EIGEN_DATA_TYPES

using namespace std;

template<typename Scalar, typename Derived>
void foo(const Eigen::Object<Scalar, Derived>& m)
{
	cout << "Here's m:" << endl << m << endl;
}

template<typename Scalar, typename Derived>
Eigen::ScalarProduct<Derived>
twice(const Eigen::Object<Scalar, Derived>& m)
{
	return 2 * m;
}

int main(int, char**)
{
	Matrix2d m;
	m(0,0)= 1;
	m(1,0)= 2;
	m(0,1)= 3;
	m(1,1)= 4;
	foo(m);
	foo(twice(m));
	return 0;
}
