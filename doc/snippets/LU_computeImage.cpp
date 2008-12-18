MatrixXd m(3,3);
m << 1,1,0,
     1,3,2,
     0,1,1;
cout << "Here is the matrix m:" << endl << m << endl;
LU<Matrix3d> lu(m);
// allocate the matrix img with the correct size to avoid reallocation
MatrixXd img(m.rows(), lu.rank());
lu.computeImage(&img);
cout << "Notice that the middle column is the sum of the two others, so the "
     << "columns are linearly dependent." << endl;
cout << "Here is a matrix whose columns have the same span but are linearly independent:"
     << endl << img << endl;
