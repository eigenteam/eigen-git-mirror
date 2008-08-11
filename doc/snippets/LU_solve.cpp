typedef Matrix<float,2,3> Matrix2x3;
typedef Matrix<float,3,2> Matrix3x2;
Matrix2x3 m = Matrix2x3::Random();
Matrix2f y = Matrix2f::Random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is the matrix y:" << endl << y << endl;
Matrix3x2 x;
if(m.lu().solve(y, &x))
{
  assert(y.isApprox(m*x));
  cout << "Here is a solution x to the equation mx=y:" << endl << x << endl;
}
else
  cout << "The equation mx=y does not have any solution." << endl;

