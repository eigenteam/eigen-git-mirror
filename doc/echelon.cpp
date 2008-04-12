#include <Eigen/Core>

USING_PART_OF_NAMESPACE_EIGEN

namespace Eigen {

/* Echelon a matrix in-place:
 *
 * Meta-Unrolled version, for small fixed-size matrices
 */
template<typename Derived, int Step>
struct unroll_echelon
{
  enum { k = Step - 1,
         Rows = Derived::RowsAtCompileTime,
         Cols = Derived::ColsAtCompileTime,
         CornerRows = Rows - k,
         CornerCols = Cols - k
  };
  static void run(MatrixBase<Derived>& m)
  {
    unroll_echelon<Derived, Step-1>::run(m);
    int rowOfBiggest, colOfBiggest;
    m.template corner<CornerRows, CornerCols>(BottomRight)
     .cwiseAbs()
     .maxCoeff(&rowOfBiggest, &colOfBiggest);
    m.row(k).swap(m.row(k+rowOfBiggest));
    m.col(k).swap(m.col(k+colOfBiggest));
    m.template corner<CornerRows-1, CornerCols>(BottomRight)
      -= m.col(k).template end<CornerRows-1>()
       * (m.row(k).template end<CornerCols>() / m(k,k));
  }
};

template<typename Derived>
struct unroll_echelon<Derived, 0>
{
  static void run(MatrixBase<Derived>& m) {}
};

/* Echelon a matrix in-place:
 *
 * Non-unrolled version, for dynamic-size matrices.
 * (this version works for all matrices, but in the fixed-size case the other
 * version is faster).
 */
template<typename Derived>
struct unroll_echelon<Derived, Dynamic>
{
  static void run(MatrixBase<Derived>& m)
  {
    for(int k = 0; k < m.diagonal().size(); k++)
    {
      int rowOfBiggest, colOfBiggest;
      int cornerRows = m.rows()-k, cornerCols = m.cols()-k;
      m.corner(BottomRight, cornerRows, cornerCols)
      .cwiseAbs()
      .maxCoeff(&rowOfBiggest, &colOfBiggest);
      m.row(k).swap(m.row(k+rowOfBiggest));
      m.col(k).swap(m.col(k+colOfBiggest));
      m.corner(BottomRight, cornerRows-1, cornerCols)
        -= m.col(k).end(cornerRows-1) * (m.row(k).end(cornerCols) / m(k,k));
    }
  }
};
using namespace std;
template<typename Derived>
void echelon(MatrixBase<Derived>& m)
{
  const int size = DiagonalCoeffs<Derived>::SizeAtCompileTime;
  const bool unroll = size <= 4;
  unroll_echelon<Derived, unroll ? size : Dynamic>::run(m);
}

template<typename Derived>
void doSomeRankPreservingOperations(MatrixBase<Derived>& m)
{
  for(int a = 0; a < 3*(m.rows()+m.cols()); a++)
  {
    double d = ei_random<double>(-1,1);
    int i = ei_random<int>(0,m.rows()-1); // i is a random row number
    int j;
    do {
      j = ei_random<int>(0,m.rows()-1);
    } while (i==j); // j is another one (must be different)
    m.row(i) += d * m.row(j);

    i = ei_random<int>(0,m.cols()-1); // i is a random column number
    do {
      j = ei_random<int>(0,m.cols()-1);
    } while (i==j); // j is another one (must be different)
    m.col(i) += d * m.col(j);
  }
}

} // namespace Eigen

using namespace std;

int main(int, char **)
{
  srand((unsigned int)time(0));
  const int Rows = 6, Cols = 6;
  typedef Matrix<double, Rows, Cols> Mat;
  const int N = Rows < Cols ? Rows : Cols;

  // start with a matrix m that's obviously of rank N-1
  Mat m = Mat::identity(Rows, Cols); // args just in case of dyn. size
  m.row(0) = m.row(1) = m.row(0) + m.row(1);

  doSomeRankPreservingOperations(m);

  // now m is still a matrix of rank N-1
  cout << "Here's the matrix m:" << endl << m << endl;

  cout << "Now let's echelon m (repeating many times for benchmarking purposes):" << endl;
  for(int i = 0; i < 1000000; i++) echelon(m);

  cout << "Now m is:" << endl << m << endl;
}
