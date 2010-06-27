MatrixXf a(2,2);
cout << "a is of size " << a.rows() << "x" << a.cols() << std::endl;
MatrixXf b(3,3);
a = b;
cout << "a is now of size " << a.rows() << "x" << a.cols() << std::endl;
