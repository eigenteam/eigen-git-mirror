Matrix3d m = Vector3d(1,2,3).asDiagonal();
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m.block(1, 1, 2, 1):" << endl << m.block(1, 1, 2, 1) << endl;
m.block(1, 0, 2, 1) = m.block(1, 1, 2, 1);
cout << "Now the matrix m is:" << endl << m << endl;
