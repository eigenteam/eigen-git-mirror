MatrixXd X = MatrixXd::Random(5,5);
MatrixXd A = X + X.transpose();
cout << "Here is a random symmetric 5x5 matrix:" << endl << A << endl << endl;

Tridiagonalization<MatrixXd> triOfA(A);
cout << "The orthogonal matrix Q is:" << endl << triOfA.matrixQ() << endl;
cout << "The tridiagonal matrix T is:" << endl << triOfA.matrixT() << endl << endl;

MatrixXd Q = triOfA.matrixQ();
MatrixXd T = triOfA.matrixT();
cout << "Q * T * Q^T = " << endl << Q * T * Q.transpose() << endl;
