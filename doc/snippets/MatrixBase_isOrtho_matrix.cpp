Matrix3d m = Matrix3d::identity();
m(0,2) = 1e-4;
cout << "Here's the matrix m:" << endl << m << endl;
cout << "m.isOrtho() returns: " << m.isOrtho() << endl;
cout << "m.isOrtho(1e-3) returns: " << m.isOrtho(1e-3) << endl;
