#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::FixedBlock<Derived, 2, 2>
topLeft2x2Corner(MatrixBase<Scalar, Derived>& m)
{
  return Eigen::FixedBlock<Derived, 2, 2>(m.ref(), 0, 0);
}

template<typename Scalar, typename Derived>
const Eigen::FixedBlock<Derived, 2, 2>
topLeft2x2Corner(const MatrixBase<Scalar, Derived>& m)
{
  return Eigen::FixedBlock<Derived, 2, 2>(m.ref(), 0, 0);
}

int main(int, char**)
{
  Matrix3d m = Matrix3d::identity();
  cout << topLeft2x2Corner(4*m) << endl; // calls the const version
  topLeft2x2Corner(m) *= 2;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
