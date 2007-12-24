Matrix4d m = Matrix4d::diagonal(Vector4d(1,2,3,4));
m.block<2, 2>(2, 0) = m.block<2, 2>(2, 2);
cout << m << endl;
