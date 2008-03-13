#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Derived>
const Eigen::CwiseUnaryOp<
  Eigen::ei_scalar_cast_op<
    typename Eigen::ei_traits<typename Derived::Scalar>::FloatingPoint
  >, Derived
>
castToFloatingPoint(const MatrixBase<Derived>& m)
{
  return m.template cast<
    typename Eigen::ei_traits<
      typename Derived::Scalar
    >::FloatingPoint
  >();
}

int main(int, char**)
{
  Matrix2i m = Matrix2i::random();
  cout << "Here's the matrix m. It has coefficients of type int."
       << endl << m << endl;
  cout << "Here's m/20:" << endl << castToFloatingPoint(m)/20 << endl;
  return 0;
}
