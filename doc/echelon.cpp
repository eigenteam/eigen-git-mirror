#include <Eigen/Core>

USING_PART_OF_NAMESPACE_EIGEN

namespace Eigen {

template<typename Derived>
void echelon(MatrixBase<Derived>& m)
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
    // important performance tip:
    // in a complex expression such as below it can be very important to fine-tune
    // exactly where evaluation occurs. The parentheses and .eval() below ensure
    // that the quotient is computed only once, and that the evaluation caused
    // by operator* occurs last.
    m.corner(BottomRight, cornerRows-1, cornerCols)
      -= m.col(k).end(cornerRows-1) * (m.row(k).end(cornerCols) / m(k,k)).eval();
  }
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
