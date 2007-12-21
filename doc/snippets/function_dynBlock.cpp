Matrix4d m = Matrix4d::identity();
m.dynBlock(2,0,2,2) = m.dynBlock(0,0,2,2);
cout << m << endl;
