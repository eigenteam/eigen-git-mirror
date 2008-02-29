Matrix4i m = Matrix4i::zero();
m.block<3,3>(1,0).setIdentity();
cout << m << endl;
