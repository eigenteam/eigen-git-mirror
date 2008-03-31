typedef Matrix3i MyMatrixType;
MyMatrixType m = MyMatrixType::random(3, 3);
cout << "Here's the matrix m:" << endl << m << endl;
typedef Eigen::Eval<Eigen::Block<MyMatrixType,1,MyMatrixType::ColsAtCompileTime> >::MatrixType MyRowType;
// now MyRowType is just the same typedef as RowVector3i
MyRowType r = m.row(0);
cout << "Here's r:" << endl << r << endl;
typedef Eigen::Eval<Eigen::Block<MyMatrixType> >::MatrixType MyBlockType;
MyBlockType c = m.corner(Eigen::TopRight, 2, 2);
// now MyBlockType is a a matrix type where the number of rows and columns
// are dynamic, but know at compile-time to be <= 2. Therefore no dynamic memory
// allocation occurs.
cout << "Here's c:" << endl << c << endl;
