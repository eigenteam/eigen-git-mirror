#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Derived>
Eigen::Block<Derived,Derived::RowsAtCompileTime,1>
firstColumn(MatrixBase<Derived>& m)
{
  return typename Eigen::Block<Derived,Derived::RowsAtCompileTime,1>(m, 0);
}

template<typename Derived>
const Eigen::Block<Derived,Derived::RowsAtCompileTime,1>
firstColumn(const MatrixBase<Derived>& m)
{
  return typename Eigen::Block<Derived,Derived::RowsAtCompileTime,1>(m, 0);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::identity();
  cout << firstColumn(2*m) << endl; // calls the const version
  firstColumn(m) *= 5;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
