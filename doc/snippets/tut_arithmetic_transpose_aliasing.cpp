Matrix2i a; a << 1, 2, 3, 4;
cout << "Here is the matrix a:\n" << a << endl;
a = a.transpose(); // fails
cout << "and the aliasing effect:\n" << a << endl;