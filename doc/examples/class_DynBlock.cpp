#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::Block<Derived>
topLeftCorner(MatrixBase<Scalar, Derived>& m, int rows, int cols)
{
  return Eigen::Block<Derived>(m.ref(), 0, 0, rows, cols);
}

template<typename Scalar, typename Derived>
const Eigen::Block<Derived>
topLeftCorner(const MatrixBase<Scalar, Derived>& m, int rows, int cols)
{
  return Eigen::Block<Derived>(m.ref(), 0, 0, rows, cols);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::identity();
  cout << topLeftCorner(4*m, 2, 3) << endl; // calls the const version
  topLeftCorner(m, 2, 3) *= 5;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
