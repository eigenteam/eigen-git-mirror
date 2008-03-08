#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

// define a custom template binary functor
template<typename Scalar>
struct CwiseClampOp EIGEN_EMPTY_STRUCT {
    CwiseClampOp(const Scalar& inf, const Scalar& sup) : m_inf(inf), m_sup(sup) {}
    Scalar operator()(const Scalar& x) const { return x<m_inf ? m_inf : (x>m_sup ? m_sup : x); }
    Scalar m_inf, m_sup;
};

int main(int, char**)
{
  Matrix4d m1 = Matrix4d::random();
  cout << m1.cwise(CwiseClampOp<Matrix4d::Scalar>(-0.5,0.5)) << endl;
  return 0;
}
