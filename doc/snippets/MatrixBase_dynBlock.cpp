Matrix3d m = Matrix3d::diagonal(Vector3d(1,2,3));
cout << "Here is the matrix m:" << endl << m << endl;
cout << "Here is m.dynBlock(1, 1, 2, 1):" << endl << m.dynBlock(1, 1, 2, 1) << endl;
m.dynBlock(1, 0, 2, 1) = m.dynBlock(1, 1, 2, 1);
cout << "Now the matrix m is:" << endl << m << endl;
