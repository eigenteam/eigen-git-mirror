Matrix4i m = Matrix4i::zero();
m.fixedBlock<3,3>(1,0).setIdentity();
cout << m << endl;
