#include "../src/Core.h"

using namespace std;

template<typename Scalar, typename Derived>
void foo(const EiObject<Scalar, Derived>& m)
{
	cout << "Here's m:" << endl << m << endl;
}

template<typename Scalar, typename Derived>
EiScalarProduct<Derived>
twice(const EiObject<Scalar, Derived>& m)
{
	return 2 * m;
}

int main(int, char**)
{
	EiMatrix2d m;
	m(0,0)= 1;
	m(1,0)= 2;
	m(0,1)= 3;
	m(1,1)= 4;
	foo(m);
	foo(twice(m));
	return 0;
}
