MatrixXf m = MatrixXf::Random(3,5);
cout << "Here is the matrix m:" << endl << m << endl;
LU<MatrixXf> lu(m);
// allocate the matrix ker with the correct size to avoid reallocation
MatrixXf ker(m.rows(), lu.dimensionOfKernel());
lu.computeKernel(&ker);
cout << "Here is a matrix whose columns form a basis of the kernel of m:"
     << endl << ker << endl;
cout << "By definition of the kernel, m*ker is zero:"
     << endl << m*ker << endl;
