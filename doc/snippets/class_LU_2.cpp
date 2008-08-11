typedef Matrix<double, 5, 3> Matrix5x3;
typedef Matrix<double, 5, 5> Matrix5x5;
Matrix5x3 m = Matrix5x3::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Eigen::LU<Matrix5x3> lu(m);
cout << "Here is, up to permutations, its LU decomposition matrix:"
     << endl << lu.matrixLU() << endl;
cout << "Here is the actual L matrix in this decomposition:" << endl;
Matrix5x5 l = Matrix5x5::Identity();
l.block<5,3>(0,0).part<StrictlyLower>() = lu.matrixLU();
cout << l << endl;
cout << "Let us now reconstruct the original matrix m:" << endl;
Matrix5x3 x = l * lu.matrixU();
Matrix5x3 y;
for(int i = 0; i < 5; i++) for(int j = 0; j < 3; j++)
  y(i, lu.permutationQ()[j]) = x(lu.permutationP()[i], j);
cout << y << endl;
assert(y.isApprox(m));
