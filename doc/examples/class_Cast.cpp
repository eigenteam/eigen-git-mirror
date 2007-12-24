#include <Eigen/Core.h>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Scalar, typename Derived>
Eigen::Cast<double, Derived>
castToDouble(const MatrixBase<Scalar, Derived>& m)
{
  return Eigen::Cast<double, Derived>(m.ref());
  // note: tempting as it is, writing "m.cast<double>()" here
  // causes a compile error with g++ 4.2, apparently due to
  // g++ getting confused by the many template types and
  // template arguments involved.
}

int main(int, char**)
{
  Matrix2i m = Matrix2i::random();
  cout << "Here's the matrix m. It has coefficients of type int."
       << endl << m << endl;
  cout << "Here's 0.05*m:" << endl << 0.05 * castToDouble(m) << endl;
  return 0;
}
