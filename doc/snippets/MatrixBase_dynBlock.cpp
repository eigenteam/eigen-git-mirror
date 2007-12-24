Matrix3d m = Matrix3d::diagonal(Vector3d(1,2,3));
m.dynBlock(1, 0, 2, 1) = m.dynBlock(1, 1, 2, 1);
cout << m << endl;
