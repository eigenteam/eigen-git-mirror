// FIXME - this example is not too good as that functionality is provided in the Eigen API
// additionally it's quite heavy. the CwiseUnaryOp example is better.

#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

// define a custom template binary functor
template<typename Scalar> struct CwiseMinOp EIGEN_EMPTY_STRUCT {
    Scalar operator()(const Scalar& a, const Scalar& b) const { return std::min(a,b); }
    enum { Cost = Eigen::ConditionalJumpCost + Eigen::NumTraits<Scalar>::AddCost };
};

// define a custom binary operator between two matrices
template<typename Derived1, typename Derived2>
const Eigen::CwiseBinaryOp<CwiseMinOp<typename Derived1::Scalar>, Derived1, Derived2>
cwiseMin(const MatrixBase<Derived1> &mat1, const MatrixBase<Derived2> &mat2)
{
  return Eigen::CwiseBinaryOp<CwiseMinOp<typename Derived1::Scalar>, Derived1, Derived2>(mat1, mat2);
}

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::random(), m2 = Matrix4d::random();
  cout << cwiseMin(m1,m2) << endl;            // use our new global operator
  cout << m1.cwise<CwiseMinOp<double> >(m2) << endl;   // directly use the generic expression member
  cout << m1.cwise(m2, CwiseMinOp<double>()) << endl; // directly use the generic expression member (variant)
  return 0;
}
