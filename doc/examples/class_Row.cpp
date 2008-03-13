#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Derived>
Eigen::Block<Derived,1,Derived::ColsAtCompileTime>
firstRow(MatrixBase<Derived>& m)
{
  return Eigen::Block<Derived,1,Derived::ColsAtCompileTime>(m, 0);
}

template<typename Derived>
const Eigen::Block<Derived,1,Derived::ColsAtCompileTime>
firstRow(const MatrixBase<Derived>& m)
{
  return Eigen::Block<Derived,1,Derived::ColsAtCompileTime>(m, 0);
}

int main(int, char**)
{
  Matrix4d m = Matrix4d::identity();
  cout << firstRow(2*m) << endl; // calls the const version
  firstRow(m) *= 5;              // calls the non-const version
  cout << "Now the matrix m is:" << endl << m << endl;
  return 0;
}
