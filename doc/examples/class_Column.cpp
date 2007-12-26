#include <Eigen/Core.h>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::Column<Derived>
firstColumn(MatrixBase<Scalar, Derived>& m)
{
  return Eigen::Column<Derived>(m.ref(), 0);
}

template<typename Scalar, typename Derived>
const Eigen::Column<Derived>
firstColumn(const MatrixBase<Scalar, Derived>& m)
{
  return Eigen::Column<Derived>(m.ref(), 0);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::identity();
  cout << firstColumn(2*m) << endl; // calls the const version
  firstColumn(m) *= 5;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
