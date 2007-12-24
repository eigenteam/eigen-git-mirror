#include <Eigen/Core.h>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::Row<Derived>
firstRow(MatrixBase<Scalar, Derived>& m)
{
  return m.row(0);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::identity();
  cout << firstRow(m) << endl;
  firstRow(m) *= 5;
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
