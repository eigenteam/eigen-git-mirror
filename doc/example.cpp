#include <Eigen/Core.h>

USING_PART_OF_NAMESPACE_EIGEN

using namespace std;

template<typename Scalar, typename Derived>
void foo(const MatrixBase<Scalar, Derived>& m)
{
  cout << "Here's m:" << endl << m << endl;
}

template<typename Scalar, typename Derived>
Eigen::ScalarMultiple<Derived>
twice(const MatrixBase<Scalar, Derived>& m)
{
  return 2 * m;
}

int main(int, char**)
{
  Matrix2d m = Matrix2d::random();
  foo(m);
  foo(twice(m));
  return 0;
}
