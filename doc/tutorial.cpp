#include <Eigen/Core>

USING_PART_OF_NAMESPACE_EIGEN

using namespace std;

int main(int, char **)
{
  Matrix<double,2,2> m; // 2x2 fixed-size matrix with uninitialized entries
  m(0,0) = 1;
  m(0,1) = 2;
  m(1,0) = 3;
  m(1,1) = 4;

  cout << "Here is a 2x2 matrix m:" << endl << m << endl;
  cout << "Let us now build a 4x4 matrix m2 by assembling together four 2x2 blocks." << endl;
  MatrixXd m2(4,4); // dynamic matrix with initial size 4x4 and uninitialized entries
  // notice how we are mixing fixed-size and dynamic-size types.
  
  cout << "In the top-left block, we put the matrix m shown above." << endl;
  m2.block<2,2>(0,0) = m;
  cout << "In the bottom-left block, we put the matrix m*m, which is:" << endl << m*m << endl;
  m2.block<2,2>(2,0) = m * m;
  cout << "In the top-right block, we put the matrix m+m, which is:" << endl << m+m << endl;
  m2.block<2,2>(0,2) = m + m;
  cout << "In the bottom-right block, we put the matrix m-m, which is:" << endl << m-m << endl;
  m2.block<2,2>(2,2) = m - m;
  cout << "Now the 4x4 matrix m2 is:" << endl << m2 << endl;
  
  cout << "Row 0 of m2 is:" << endl << m2.row(0) << endl;
  cout << "The third element in that row is " << m2.row(0)[2] << endl;
  cout << "Column 1 of m2 is:" << endl << m2.col(1) << endl;
  cout << "The transpose of m2 is:" << endl << m2.transpose() << endl;
  cout << "The matrix m2 with row 0 and column 1 removed is:" << endl << m2.minor(0,1) << endl;
  return 0;
}
