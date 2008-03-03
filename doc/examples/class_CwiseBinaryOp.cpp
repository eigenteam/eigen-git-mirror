#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

// define a custom template binary functor
struct CwiseMinOp {
    template<typename Scalar>
    static Scalar op(const Scalar& a, const Scalar& b) { return std::min(a,b); }
};

// define a custom binary operator between two matrices
template<typename Scalar, typename Derived1, typename Derived2>
const Eigen::CwiseBinaryOp<CwiseMinOp, Derived1, Derived2>
cwiseMin(const MatrixBase<Scalar, Derived1> &mat1, const MatrixBase<Scalar, Derived2> &mat2)
{
  return Eigen::CwiseBinaryOp<CwiseMinOp, Derived1, Derived2>(mat1.ref(), mat2.ref());
  // Note that the above is equivalent to:
  // return mat1.template cwise<CwiseMinOp>(mat2);
}

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::random(), m2 = Matrix4d::random();
  cout << cwiseMin(m1,m2) << endl;          // use our new global operator
  cout << m1.cwise<CwiseMinOp>(m2) << endl; // directly use the generic expression member
  return 0;
}
