typedef Matrix<double, 5, 3> Matrix5x3;
typedef Matrix<double, 5, 5> Matrix5x5;
Matrix5x3 m = Matrix5x3::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Eigen::FullPivLU<Matrix5x3> lu(m);
cout << "Here is, up to permutations, its LU decomposition matrix:"
     << endl << lu.matrixLU() << endl;
cout << "Here is the L part:" << endl;
Matrix5x5 l = Matrix5x5::Identity();
l.block<5,3>(0,0).part<StrictlyLowerTriangular>() = lu.matrixLU();
cout << l << endl;
cout << "Here is the U part:" << endl;
Matrix5x3 u = lu.matrixLU().part<UpperTriangular>();
cout << u << endl;
cout << "Let us now reconstruct the original matrix m:" << endl;
Matrix5x3 x = l * u;
Matrix5x3 y;
for(int i = 0; i < 5; i++) for(int j = 0; j < 3; j++)
  y(i, lu.permutationQ()[j]) = x(lu.permutationP()[i], j);
cout << y << endl; // should be equal to the original matrix m
