Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Eigen::LU<Matrix3d> lu(m);
cout << "Here is, up to permutations, its LU decomposition matrix:"
     << endl << lu.matrixLU() << endl;
cout << "Let us now reconstruct the original matrix m from it:" << endl;
Matrix3d x = lu.matrixL() * lu.matrixU();
Matrix3d y;
for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
  y(i, lu.permutationQ()[j]) = x(lu.permutationP()[i], j);
cout << y << endl;
assert(y.isApprox(m));
