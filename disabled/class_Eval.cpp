#include <Eigen/Core>
USING_PART_OF_NAMESPACE_EIGEN
using namespace std;

template<typename Derived>
const Eigen::Eval<Eigen::Transpose<Derived> >
evaluatedTranspose(const MatrixBase<Derived>& m)
{
  return m.transpose().eval();
}

int main(int, char**)
{
  Matrix2f M = Matrix2f::random();
  Matrix2f m;
  m = M;
  cout << "Here is the matrix m:" << endl << m << endl;
  cout << "Now we want to replace m by its own transpose." << endl;
  cout << "If we do m = m.transpose(), then m becomes:" << endl;
  m = m.transpose();
  cout << m << endl << "which is wrong!" << endl;
  cout << "Now let us instead do m = evaluatedTranspose(m). Then m becomes" << endl;
  m = M;
  m = evaluatedTranspose(m);
  cout << m << endl << "which is right." << endl;

  return 0;
}
