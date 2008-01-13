Matrix4i m = Matrix4i::random();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m.block(1, 1, 2, 2):" << endl << m.block(1, 1, 2, 2) << endl;
m.block(1, 1, 2, 2).setZero();
cout << "Now the matrix m is:" << endl << m << endl;
