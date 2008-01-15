#include <Eigen/Core>

USING_PART_OF_NAMESPACE_EIGEN

namespace Eigen {

template<typename Scalar, typename Derived>
void echelon(MatrixBase<Scalar, Derived>& m)
{
  const int N = std::min(m.rows(), m.cols());

  for(int k = 0; k < N; k++)
  {
    int rowOfBiggest, colOfBiggest;
    int cornerRows = m.rows()-k;
    int cornerCols = m.cols()-k;
    m.corner(BottomRight, cornerRows, cornerCols)
     .findBiggestCoeff(&rowOfBiggest, &colOfBiggest);
    m.row(k).swap(m.row(k+rowOfBiggest));
    m.col(k).swap(m.col(k+colOfBiggest));
    for(int r = k+1; r < m.rows(); r++)
      m.row(r).end(cornerCols) -= m.row(k).end(cornerCols) * m(r,k) / m(k,k);
  }
}

template<typename Scalar, typename Derived>
void doSomeRankPreservingOperations(MatrixBase<Scalar, Derived>& m)
{
  for(int a = 0; a < 3*(m.rows()+m.cols()); a++)
  {
    double d = Eigen::random<double>(-1,1);
    int i = Eigen::random<int>(0,m.rows()-1); // i is a random row number
    int j;
    do {
      j = Eigen::random<int>(0,m.rows()-1);
    } while (i==j); // j is another one (must be different)
    m.row(i) += d * m.row(j);

    i = Eigen::random<int>(0,m.cols()-1); // i is a random column number
    do {
      j = Eigen::random<int>(0,m.cols()-1);
    } while (i==j); // j is another one (must be different)
    m.col(i) += d * m.col(j);
  }
}

} // namespace Eigen

using namespace std;

int main(int, char **)
{
  srand((unsigned int)time(0));
  const int Rows = 6, Cols = 4;
  typedef Matrix<double, Rows, Cols> Mat;
  const int N = Rows < Cols ? Rows : Cols;

  // start with a matrix m that's obviously of rank N-1
  Mat m = Mat::identity(Rows, Cols); // args just in case of dyn. size
  m.row(0) = m.row(1) = m.row(0) + m.row(1);

  doSomeRankPreservingOperations(m);

  // now m is still a matrix of rank N-1
  cout << "Here's the matrix m:" << endl << m << endl;

  cout << "Now let's echelon m:" << endl;
  echelon(m);

  cout << "Now m is:" << endl << m << endl;
}
