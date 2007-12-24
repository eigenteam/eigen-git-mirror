#include <Eigen/Core.h>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::Block<Derived, 2, 2>
topLeft2x2Corner(MatrixBase<Scalar, Derived>& m)
{
  return Eigen::Block<Derived, 2, 2>(m.ref(), 0, 0);
  // note: tempting as it is, writing "m.block<2,2>(0,0)" here
  // causes a compile error with g++ 4.2, apparently due to
  // g++ getting confused by the many template types and
  // template arguments involved.
}

int main(int, char**)
{
  Matrix3d m = Matrix3d::identity();
  cout << topLeft2x2Corner(m) << endl;
  topLeft2x2Corner(m) *= 2;
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
